""" 
This script is used for inference.

Variable conventions:
-- we can infer publications with whatever id they have as long as they have the required metadata
-- however the id will be called "doi" in the code
"""


import os
import pickle
import json
import argparse
import logging

from pprint import pprint
from tqdm import tqdm
from collections import Counter
from venue_parser import VenueParser
from multigraph import MultiGraph
from utils import TextProcessor


def infer_relationship(entity, multigraph, top_L1, top_L2, top_L3, top_L4, overwrite, relationship):

        # multigraph.infer_layer(entity_chain=["doi", "venue", "L4"], relationship_chain=[relationship, "in_L4"],
        #                        overwrite=overwrite, max_links=top_L4)
        # multigraph.infer_layer(entity_chain=["doi", "venue", "L3"], relationship_chain=[relationship, "in_L3"],
        #                        overwrite=overwrite, max_links=top_L3)
        # multigraph.infer_layer(entity_chain=["doi", "venue", "L2"], relationship_chain=[relationship, "in_L2"],
        #                        overwrite=overwrite, max_links=top_L2)
        # multigraph.infer_layer(entity_chain=["doi", "venue", "L1"], relationship_chain=[relationship, "in_L1"],
        #                        overwrite=overwrite, max_links=top_L1)

        multigraph.infer_layer(entity_chain=["doi", "venue", "L4"], relationship_chain=[relationship, "in_L4"],
                               overwrite=overwrite, max_links=top_L4)
        multigraph.infer_layer(entity_chain=["doi", "venue", "L3"], relationship_chain=[relationship, "in_L3"],
                               overwrite=overwrite, max_links=top_L3)
        multigraph.infer_layer(entity_chain=["doi", "venue", "L2"], relationship_chain=[relationship, "in_L2"],
                               overwrite=overwrite, max_links=top_L2)
        multigraph.infer_layer(entity_chain=["doi", "venue", "L1"], relationship_chain=[relationship, "in_L1"],
                               overwrite=overwrite, max_links=top_L1)


def add_to_predictions(tups, title, abstract):
    # infer L5/L6 for the inferred L4s
    processed_title = text_processor.preprocess_text(title)
    processed_abstract = text_processor.preprocess_text(abstract)
    my_text = processed_title + processed_abstract
    final_tups = []
    for tup in tups:
        # check if we have an L4
        if len(tup) > 3:
            # this tup has an inferred L4 -- infer its L5/L6
            l4 = tup[3][0]
            my_l5s = [node[0] for node in multigraph.nodes(data='L5') if node[1] and l4 in node[0]]
            l5s = get_l5_filtering_embeddings(
                my_text, my_l5s, False, 5, text_processor, multigraph, only_text=False
            )
            if l5s:
                # check if we have inferred l5s and add them to the tuples
                final_tups.extend([(tup[0], tup[1], tup[2], tup[3], (l5[0], l5[1]), '/'.join(list(l5[2]))) for l5 in l5s])
            else:
                # no l5s were inferred -- add None to l5 and l6
                to_add = list(tup)
                to_add.extend([None, None])
                final_tups.append(tuple(to_add))
        else:
            # this tup does not have an inferred L4 -- add None
            to_add = list(tup)
            to_add.extend([None, None, None])
            final_tups.append(tuple(to_add))
    return final_tups


def infer_l5_l6(tups, doi, title, abstract):
    # get the inferred L4s and infer their L5/L6
    # check if the title and abstract are available
    if title != '' and abstract != '':
        preds = add_to_predictions(tups, title, abstract)
    elif title != '' and abstract == '':
        logger.info(f'No abstract available for {doi}, infer only with title')
        preds = add_to_predictions(tups, title, '')
    elif title == '' and abstract != '':
        logger.info(f'No title available for {doi}, infer only with abstract')
        preds = add_to_predictions(tups, '', abstract)
    else:
        # both are empty
        logger.info(f'No title and abstract available for {doi}')
        preds = add_to_predictions(tups, '', '')
    return preds


def infer(**kwargs):
    # defaults
    top_L1 = kwargs.get('top_L1', 1)
    top_L2 = kwargs.get('top_L2', 2)
    top_L3 = kwargs.get('top_L3', 3)
    top_L4 = kwargs.get('top_L4', 4)
    # other variables
    emphasize = kwargs.get('emphasize', 'citations')
    # return_triplets = kwargs.get('return_triplets', True)
    # ids of publications
    ids = kwargs['payload']['dois']
    published_venues = kwargs['payload']['published_venues']
    cit_ref_venues = kwargs['payload']['cit_ref_venues']
    titles = kwargs['payload']['titles']
    abstracts = kwargs['payload']['abstracts']
    # add the publications to the graph that we are going to infer
    logger.info('Adding publications to the graph')
    add(multigraph, published_venues, cit_ref_venues)
    # inferring relationships
    logger.info('Inferring relationships')
    _ = [
        infer_relationship(ids, multigraph, top_L1, top_L2, top_L3, top_L4, overwrite=True,
                           relationship='cites'),
        infer_relationship(ids, multigraph, top_L1, top_L2, top_L3, top_L4, overwrite=False,
                           relationship='published')] if emphasize == 'citations' else [
        infer_relationship(ids, multigraph, top_L4, overwrite=True,
                           relationship='published'),
        infer_relationship(ids, multigraph, top_L4, overwrite=False,
                           relationship='cites')]

    out = {}
    logger.info('Retrieving results for publications')
    for doi in tqdm(ids, desc='Retrieving results for publications'):
        L1 = [(relationship[1], relationship[2]) for relationship in multigraph.edges(data='in_L1', nbunch=doi) if
            relationship[2]]
        L2 = [(relationship[1], relationship[2]) for relationship in multigraph.edges(data='in_L2', nbunch=doi) if
            relationship[2]]
        L3 = [(relationship[1], relationship[2]) for relationship in multigraph.edges(data='in_L3', nbunch=doi) if
            relationship[2]]
        L4 = [(relationship[1], relationship[2]) for relationship in multigraph.edges(data='in_L4', nbunch=doi) if
            relationship[2]]
        """ This is for not returning triplets -- we comment it out """
        """
        if not return_triplets:
            # check if we only inferred L4 -- it is possible
            if not L3 and not L2 and not L1:
                my_triplets = [(L2_to_L1[list(L3_to_L2[L4_to_L3[tup[0]]].keys())[0]], list(L3_to_L2[L4_to_L3[tup[0]]].keys())[0], L4_to_L3[tup[0]], tup) for tup in L4]
                out[doi] = [
                    (
                        {'L1': triplet[0], 'L2': triplet[1], 'L3':triplet[2], 'L4': triplet[3][0], 'score_for_L4': triplet[3][1]}
                    ) for triplet in my_triplets
                ]
            else:
                out[doi] = {
                    'L1': L1,
                    'L2': L2,
                    'L3': L3,
                    'L4': L4
                }
        else:"""
        ############################################
        # check if we only inferred L4 -- it is possible
        if not L3 and not L2 and not L1:
            my_triplets = [(L2_to_L1[list(L3_to_L2[L4_to_L3[tup[0]]].keys())[0]],
                            list(L3_to_L2[L4_to_L3[tup[0]]].keys())[0], L4_to_L3[tup[0]], tup) for tup in L4]
            ############################################
            # infer L5 and L6
            preds_with_l5_l6 = infer_l5_l6(my_triplets, doi, titles[doi], abstracts[doi])
            out[doi] = [
                (
                    {
                        'L1': triplet[0],
                        'L2': triplet[1],
                        'L3': triplet[2],
                        'L4': triplet[3][0] if triplet[3] else None,
                        'L5': triplet[4][0] if triplet[4] else None,
                        'L6': triplet[5] if triplet[5] else None,
                        'score_for_L4': triplet[3][1] if triplet[3] else 0.0,
                        'score_for_L5': triplet[4][1] if triplet[4] else 0.0,
                    }
                ) for triplet in preds_with_l5_l6
            ]
            ############################################
        else:
            l3_mapping_to_l2 = [(tup, list(L3_to_L2[tup[0]].keys())) for tup in L3]
            flatten_l3_to_l2 = [(tup[0], l2) for tup in l3_mapping_to_l2 for l2 in tup[1]]
            l2_mapping_to_l1 = [(tup[0], tup[1], L2_to_L1[tup[1]]) for tup in flatten_l3_to_l2]

            if L4:
                filtered_l4 = [
                    (l4, (L3[[l3[0] for l3 in L3].index(L4_to_L3[l4[0]])])) for l4 in L4 if
                            L4_to_L3[l4[0]] in [l3[0] for l3 in L3]
                ]
                my_tups = []
                for tup in l2_mapping_to_l1:
                    if tup[0][0] in [i[1][0] for i in filtered_l4]:
                        my_tups.append((tup[2], tup[1], tup[0], filtered_l4[[i[1][0] for i in filtered_l4].index(tup[0][0])][0]))
                    else:
                        my_tups.append((tup[2], tup[1], tup[0]))
                ############################################
                # infer the L5 and L6
                preds_with_l5_l6 = infer_l5_l6(
                    my_tups,
                    doi,
                    titles[doi],
                    abstracts[doi]
                )
                ############################################
                out[doi] = [
                    (
                        {
                            'L1': triplet[0],
                            'L2': triplet[1],
                            'L3': triplet[2][0],
                            'L4': triplet[3][0] if triplet[3] else None,
                            'L5': triplet[4][0] if triplet[4] else None,
                            'L6': triplet[5] if triplet[5] else None,
                            'score_for_L3': triplet[2][1] if triplet[2] else 0.0,
                            'score_for_L4': triplet[3][1] if triplet[3] else 0.0,
                            'score_for_L5': triplet[4][1] if triplet[4] else 0.0
                        }
                    ) for triplet in preds_with_l5_l6
                ]
            else:
                out[doi] = [
                    (
                        {
                            'L1': triplet[2],
                            'L2': triplet[1],
                            'L3': triplet[0][0],
                            'L4': None,
                            'L5': None,
                            'L6': None,
                            'score_for_L3': triplet[0][1], 
                            'score_for_L4': 0.0, 
                            'score_for_L5': 0.0
                        }
                    ) for triplet in l2_mapping_to_l1
                ]
    ########################################
    # clean the graph from the dois that where inferred
    logger.info('Cleaning the graph from the inferred nodes')
    multigraph.remove_nodes_from(ids)
    ########################################
    return out


def add(multigraph, published_venues, cit_ref_venues):
    multigraph.add_entities(from_entities="doi", to_entities="venue", relationship_type="published",
                            relationships=published_venues)
    multigraph.add_entities(from_entities="doi", to_entities="venue", relationship_type="cites",
                            relationships=cit_ref_venues)


def one_ranking(l5s_to_keep, canditate_l5, my_occurences, my_graph):
    final_ranking = []
    l4_to_l5 = dict()
    for l5, kws in l5s_to_keep.items():
        # get l4
        l4 = '_'.join(l5.split('_')[:-1])
        if l4 not in l4_to_l5:
            l4_to_l5[l4] = []
            l4_to_l5[l4].append((l5, kws))
        else:
            l4_to_l5[l4].append((l5, kws))
    my_counter = dict()
    for l4, l5s in l4_to_l5.items():
        my_counter[l4] = len(l5s)
    # sorted counter by num of occurences
    sorted_counter = {key: l4_to_l5[key] for key in sorted(l4_to_l5, key=lambda x: len(l4_to_l5[x]), reverse=True)}
    # sort the values by the number of words in each l5
    for key, value in sorted_counter.items():
        sorted_counter[key] = sorted(value, key=lambda x: len(x[1]), reverse=True)
    # normalize the counter
    factor=1.0/sum(my_counter.values())
    my_counter = {key: value*factor for key, value in my_counter.items()}
    for l4, l5s in sorted_counter.items():
        for l5, kws in l5s:
            if l5 not in canditate_l5:
                continue
            total_score = sum([my_occurences[kw] * my_counter[l4] for kw in kws]) * len(kws)
            words_score = sum([my_occurences[kw] * my_graph[kw][l5][0]['in_L5'] for kw in kws])
            final_ranking.append((l5, total_score, words_score, kws))
    #########################################   
    # the above method produces a lot of level 5s under the same level 4 with the same score..re-rank them according 
    # to the score of each word in the l5
    sorted_final_ranking = sorted(final_ranking, key=lambda x: x[1], reverse=True)
    sorted_final_ranking_per_l4 = dict()
    for tup in sorted_final_ranking:
        l4 = '_'.join(tup[0].split('_')[:-1])
        if l4 not in sorted_final_ranking_per_l4:
            sorted_final_ranking_per_l4[l4] = []
            sorted_final_ranking_per_l4[l4].append(tup)
        else:
            sorted_final_ranking_per_l4[l4].append(tup)

    return [(l4, sorted(l5s, key= lambda x: x[2], reverse=True)[0], list(set([l1 for l in l5s for l1 in l[3]]))) for l4, l5s in sorted_final_ranking_per_l4.items()]


def get_l5_filtering_embeddings(abstract, canditate_l5, preprocess, topk, text_processor, my_graph, only_text=False):
    if preprocess:
        pre_abstract = text_processor.preprocess_text(abstract)
    else:
        pre_abstract = abstract
    if abstract == '' or abstract is None:
        return []
    trigrams = text_processor.get_ngrams(pre_abstract, k=3)
    bigrams = text_processor.get_ngrams(pre_abstract, k=2)
    unigrams = text_processor.get_ngrams(pre_abstract, k=1)

    # the bigrams and trigrams that are identical will also be in the hits
    bigram_hits = text_processor.retrieve_similar_nodes(bigrams, topk)
    bigram_hits = set([b for bi in bigram_hits if bi for tup in bi for b in tup])

    # this also returns bigrams -- maybe limit only to trigrams
    trigram_hits = text_processor.retrieve_similar_nodes(trigrams, topk)    
    trigram_hits = set([b for bi in trigram_hits if bi for tup in bi for b in tup])

    candidates = []
    candidates.extend(
        [(bi, [n for n in my_graph[bi] if 'L5' in my_graph.nodes[n]]) for bi in bigram_hits if bi in my_graph])
    candidates.extend(
        [(bi, [n for n in my_graph[bi] if 'L5' in my_graph.nodes[n]]) for bi in trigram_hits if bi in my_graph])
    
    # we use unigrams as is -- we do not want to add more noise
    candidates.extend(
        [(bi, [n for n in my_graph[bi] if 'L5' in my_graph.nodes[n]]) for bi in unigrams if bi in my_graph])
    
    words = []
    word_to_l5s = dict()
    for cand in candidates:
        words.append(cand[0])
        for c in cand[1]:
            try:
                word_to_l5s[c].add(cand[0])
            except KeyError:
                word_to_l5s[c] = {cand[0]}

    the_occurences = Counter(words)
    l5s_to_keep = {k: v for k, v in word_to_l5s.items() if len(v) > 1}
    if not l5s_to_keep:
        return []
    
    if only_text:
        # only for text
        #########################################
        results = one_ranking(l5s_to_keep, canditate_l5, the_occurences)
        # to the resulting L4 above the L5 add all the other words that match the L5s
        # get the words that we want to keep
        to_return_top_res = [(r[1][0], r[1][1], r[1][3], r[2]) for r in results[:2]]
        words_matched = set([w for r in to_return_top_res for w in r[3]])
        rest_of_words_to_check = set([kw for _, kws in l5s_to_keep.items() for kw in kws]).difference(words_matched)
        # remove from l5s_to_keep the l3s that we already have
        second_l5s_to_keep = {l5: kws.intersection(rest_of_words_to_check) for l5, kws in l5s_to_keep.items() if '_'.join(l5.split('_')[:-2]) not in ['_'.join(r[0].split('_')[:-2]) for r in to_return_top_res]}
        # do another cycle of ranking
        if second_l5s_to_keep:
            results2 = one_ranking(second_l5s_to_keep, canditate_l5, the_occurences)
            to_return_top_res.extend([(r[1][0], r[1][1], r[1][3], r[2]) for r in results2[:1]])
        return to_return_top_res
    else:
        final_ranking = []
        for l5, kws in l5s_to_keep.items():
            if l5 not in canditate_l5:
                continue
            total_score = 0
            for kw in kws:
                sc = my_graph[kw][l5][0]['in_L5']
                total_score += the_occurences[kw] * sc
            final_ranking.append((l5, total_score, kws))
        my_res = sorted(final_ranking, key=lambda x: x[1], reverse=True)[:3]
        return my_res


def load_flatten_words_top_topic(topics_dir):
    sectors_to_words = dict()
    for file in os.listdir(topics_dir):

        if 'all_topics' not in file:
            continue

        with open(os.path.join(topics_dir, file), 'rb') as fin:
            my_words = pickle.load(fin)

        # convert to l5_dict
        for l5, data in my_words.items():
            l5 = l5.replace('@', '&')
            l4_id = l5.split('_')[-2]
            l5_id = l5.split('_')[-1]
            l5 = ' '.join(l5.split('_')[1:-2])
            l5 = f'L4_{l5}_{l4_id}_{l5_id}'
            sectors_to_words[l5] = data

    return sectors_to_words


def test():
    # initializations
    my_venue_parser = VenueParser(abbreviation_dict='venues_maps.p')
    multigraph = MultiGraph('scinobo_inference_graph.p')
    text_processor = TextProcessor()
    my_title = """Embedding Biomedical Ontologies by Jointly Encoding Network Structure and Textual Node Descriptors"""
    my_abstract = """Network Embedding (NE) methods, which
    map network nodes to low-dimensional feature vectors, have wide applications in network analysis and bioinformatics. Many existing NE methods rely only on network structure, overlooking other information associated
    with the nodes, e.g., text describing the nodes.
    Recent attempts to combine the two sources of
    information only consider local network structure. We extend NODE2VEC, a well-known NE
    method that considers broader network structure, to also consider textual node descriptors
    using recurrent neural encoders. Our method
    is evaluated on link prediction in two networks derived from UMLS. Experimental results demonstrate the effectiveness of the proposed approach compared to previous work."""

    payload = {
        "doi": "10.18653/v1/w19-5032",
        "doi_cites_venues": {
            "10.18653/v1/w19-5032": {
                "acl": 2,
                "aimag": 1,
                "arxiv artificial intelligence": 1,
                "arxiv computation and language": 2,
                "arxiv machine learning": 1,
                "arxiv social and information networks": 1,
                "briefings in bioinformatics": 1,
                "comparative and functional genomics": 1,
                "conference of the european chapter of the association for computational linguistics": 1,
                "cvpr": 1,
                "emnlp": 3,
                "eswc": 1,
                "iclr": 2,
                "icml": 1,
                "ieee trans signal process": 1,
                "j mach learn res": 1,
                "kdd": 4,
                "naacl": 1,
                "nips": 1,
                "nucleic acids res": 1,
                "pacific symposium on biocomputing": 3,
                "physica a statistical mechanics and its applications": 1,
                "proceedings of the acm conference on bioinformatics computational biology and health informatics": 1,
                "sci china ser f": 1,
                "the web conference": 1
            }
        },
        "doi_publish_venue": {
            "10.18653/v1/w19-5032": {
                "proceedings of the bionlp workshop and shared task": 1
            }
        },
        "emphasize": "citations"
    }

    my_res = infer(payload, multigraph, my_venue_parser)
    my_l4 = my_res["10.18653/v1/w19-5032"][0]['L4'] if 'L4' in my_res["10.18653/v1/w19-5032"][0] else None
    if my_l4 is None:
        return
    title = text_processor.preprocess_text(my_title)
    abstract = text_processor.preprocess_text(my_abstract)
    my_text = title + ' ' + abstract
    my_l5s = [node[0] for node in multigraph.nodes(data='L5') if node[1] and my_l4 in node[0]]
    l5s = get_l5_filtering_embeddings(
        my_text, my_l5s, False, 5, text_processor, multigraph, only_text=False
    )
    pprint(l5s)


def parse_args():
    ##############################################
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, default='/input_files', help="The directory where the chunks of publications along with metadata exist", required=False)
    parser.add_argument("--out_path", type=str, default='/output_files', help="The directory where the output files will be written", required=False)
    parser.add_argument("--log_path", type=str,default='/output_files/fos_inference.log',  help="The path for the log file.", required=False)
    parser.add_argument("--emphasize", type=str,default='citations',  help="If you want to emphasize in published venue or the cit/refs", required=False)
    # parser.add_argument("--return_triplets", type=bool,default=True,  help="If you want to enforce hierarchy", required=False)
    parser.add_argument("--batch_size", type=int, default=500,  help="The batch size", required=False)
    args = parser.parse_args()
    return args
    ##############################################


def yielder(input_dir):
    for file in os.listdir(input_dir):
        if file.endswith('.jsonl'):
            with open(os.path.join(input_dir, file), 'r') as fin:
                yield json.loads(fin.readline().strip()), file
    

def yielder_json(input_dir):
    for file in os.listdir(input_dir):
        if file.endswith('.json'):
            with open(os.path.join(input_dir, file), 'r') as fin:
                yield json.load(fin), file
    

def create_payload(dato):
    payload = {
        'dois': [],
        'cit_ref_venues': {},
        'published_venues': {},
        'titles': {},
        'abstracts': {}
    }
    for d in tqdm(dato, desc='Creating payload for inference'):
        # input checks
        if 'id' not in d:
            logger.info('publication without id, skipping ...')
            continue
        my_id = d['id']
        if 'cit_venues' not in d and 'ref_venues' not in d:
            logger.info(f'publication with {my_id} has no cit_venues and ref_venues, skipping ...')
            continue
        elif 'cit_venues' in d and 'ref_venues' not in d:
            cit_venues = d['cit_venues']
        elif 'cit_venues' not in d and 'ref_venues' in d:
            ref_venues = d['ref_venues']
        else:
            cit_venues = d['cit_venues']
            ref_venues = d['ref_venues']
        if 'pub_venue' not in d:
            logger.info(f'publication with {my_id} has no pub_venue, leaving this field empty')
            pub_venue = ''
        else:
            pub_venue_res = venue_parser.preprocess_venue(d['pub_venue'])
            if pub_venue_res is None:
                pub_venue = ''
            else:
                pub_venue, _ = pub_venue_res[0], pub_venue_res[1]
        # preprocess the venues
        ##############################################
        counts_ref = []
        counts_cit = []
        for ven in cit_venues:
            res = venue_parser.preprocess_venue(ven)
            if res is None:
                continue
            else:
                pre_ven, _ = res[0], res[1]
            counts_cit.append(pre_ven)
        for ven in ref_venues:
            res = venue_parser.preprocess_venue(ven)
            if res is None:
                continue
            else:
                pre_ven, _ = res[0], res[1]
            counts_ref.append(pre_ven)
        ##############################################
        counts_ref = Counter([ven for ven in counts_ref if ven is not None]).most_common()
        counts_cit = Counter([ven for ven in counts_cit if ven is not None]).most_common()
        ##############################################
        weighted_referenced_venues = {}
        for c in counts_ref:
            weighted_referenced_venues[c[0]] = c[1]

        for c in counts_cit:
            try:
                weighted_referenced_venues[c[0]] += c[1]
            except KeyError:
                weighted_referenced_venues[c[0]] = c[1]
        ##############################################
        payload['cit_ref_venues'][my_id] = weighted_referenced_venues
        ##############################################
        if pub_venue != '':
            payload['published_venues'][my_id] = {pub_venue: 1}
        else:
            payload['published_venues'][my_id] = {}
        ##############################################
        payload['dois'].append(my_id)
        if 'title' not in d:
            payload['titles'][my_id] = ''
        else:
            payload['titles'][my_id] = d['title']
        if 'abstract' not in d:
            payload['abstracts'][my_id] = ''
        else:
            payload['abstracts'][my_id] = d['abstract']
    return payload


if __name__ == '__main__':
    # parse the arguments
    arguments = parse_args()
    # init the logger
    logging.basicConfig(
        filename=arguments.log_path,
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )
    logging.info("Running FoS inference")
    logger = logging.getLogger('inference')
    # create the output directory
    logger.info('Creating the output directory: {}'.format(arguments.out_path))
    os.makedirs(arguments.out_path, exist_ok=True)
    logger.info('Output directory created: {}'.format(arguments.out_path))
    # check if the input directory exists
    logger.info('Checking if the input directory exists: {}'.format(arguments.in_path))
    if not os.path.exists(arguments.in_path):
        logger.error('The input directory does not exist: {}'.format(arguments.in_path))
        raise Exception('The input directory does not exist: {}'.format(arguments.in_path))
    # rest of initializations
    logger.info('Initializing the venue parser')
    venue_parser = VenueParser(abbreviation_dict='venues_maps.p')
    logger.info('Initializing the multigraph')
    multigraph = MultiGraph('scinobo_inference_graph.p')
    logger.info('Initializing the text processor')
    text_processor = TextProcessor()
    # load mapping of the texonomy
    logger.info('Loading the mappings of the taxonomy')
    # load mappings
    with open('L2_to_L1.json', 'r') as fin:
        L2_to_L1 = json.load(fin)
    with open('L3_to_L2.json', 'r') as fin:
        L3_to_L2 = json.load(fin)
    with open('L4_to_L3.json', 'r') as fin:
        L4_to_L3 = json.load(fin)
    # make sure that the publications have the necessary metadata
    total_files = len([f for f in os.listdir(arguments.in_path) if f.endswith('.json')])
    batch_size = arguments.batch_size
    for idx, tup in enumerate(tqdm(yielder_json(arguments.in_path), desc='Parsing input files for inference', total=total_files)):
        dato, file_name = tup[0], tup[1]
        # each dato has lines of publications
        # split the lines into chunks
        chunks = [dato[i:i + batch_size] for i in range(0, len(dato), batch_size)]
        chunk_predictions = []
        # parse the chunks -- for each chunk create the payload for inference
        logger.info(f'Inferring chunks of file number:{idx} and file name: {file_name}')
        for chunk in tqdm(chunks, desc=f'Inferring chunks of file number:{idx} and file name: {file_name}'):
            logger.info('Creating payload for chunk')
            payload_to_infer = create_payload(chunk)
            logger.info('Payload for chunk')
            # infer to Level 1 - Level 4
            logger.info('Inferring up to Level 6')
            infer_res = infer(
                emphasize=arguments.emphasize,
                # return_triplets=arguments.return_triplets,
                payload = payload_to_infer
            )
            logger.info(f'Inference up to Level 6 done for chunk')
            res_to_dump = [
                {
                    'id': k, 
                    'fos_predictions': [
                        {
                            'Level 1': pr['L1'], 
                            'Level 2': pr['L2'], 
                            'Level 3': pr['L3'], 
                            'Level 4': pr['L4'], 
                            'Level 5': pr['L5'], 
                            'Level 6': pr['L6']    
                        } for pr in v
                    ],
                    'fos_scores': [
                        {
                            'score_for_level_3': pr['score_for_L3'],
                            'score_for_level_4': pr['score_for_L4'],
                            'score_for_level_5': pr['score_for_L5']
                        } for pr in v
                    ]
                } for k, v in infer_res.items()
            ]
            chunk_predictions.extend(res_to_dump)
        # dump the predictions
        logger.info(f'Dumping the predictions for the file with index: {idx} and file name: {file_name}')
        output_file_name = os.path.join(arguments.out_path, file_name)
        with open(output_file_name, 'w') as fout:
            json.dump(chunk_predictions, fout)
