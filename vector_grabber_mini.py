from wvlib import wvlib
import numpy
import random
from scipy.spatial.distance import cosine

def get_data_with_negs(limit=1000000, output=True):

    #limit = 1000
    examples, vocab_list, vecs = get_data(limit=limit)
    #Let us create negative examples
    n_examples = []
    e_examples = []

    objects = []

    for e in examples:
        if e[-1].startswith('X'):
            n_examples.append(e)
            objects.append(e[0][-1])
        else:
            e_examples.append(e)

    i_examples = []
    random.seed(1234)
    random.shuffle(objects)
    for e in n_examples:
        original_object = e[0][2]
        original_vector = vecs[original_object]

        rnd_obj = random.choice(objects)
        #import pdb;pdb.set_trace()
        while cosine(vecs[rnd_obj], original_vector) < 0.6:
            rnd_obj = random.choice(objects)

        i_examples.append(((e[0][0], e[0][1], rnd_obj), 'IX'))

    #Txt output it!
    out = open('dataset_xm.txt', 'wt')
    for l in [e_examples, n_examples, i_examples]:
        for e in l:
            try:
                tokens = [vocab_list[e[0][0]], vocab_list[e[0][1]], vocab_list[e[0][2]], e[-1]]
                out.write('\t'.join(tokens) + '\n')
            except:
                print l
                pass#import pdb;pdb.set_trace()

    out.close()
    #Pickle it out!
    import pickle
    out = open('dataset_xm.list', 'wb')
    pickle.dump([n_examples + e_examples + i_examples, vocab_list, vecs], out)
    out.close()




def get_data(limit=100000):


    #Let us say the point of this little program is to get the data,
    #get the vectors and the create both embedding matrix
    #and the data in nice indexes

    #wv = wvlib.load("/usr/share/ParseBank/vector-space-models/FIN/w2v_pbv3_lm.rev01.bin",max_rank=1000000)
    wv = wvlib.load("/home/ginter/w2v/pb34_lemma_200_v2.bin").normalize()#,max_rank=10000000000).normalize()
    #wv.normalize()
    #remember to normalize!
    lines2 = open('./example_harvest/the_res', 'rt').readlines()[:50]
    lines = open('test_sent_2.txt', 'rt').readlines()

    lines += lines2

    #Such a small vocab I can ignore this stuff: vocab_set = set()
    vocab_list = []
    vecs = []

    examples = []
    corrupt_examples = []

    labels = []
    incomplete = []

    triplets = set()

    for line in lines[1:]:
        if len(line) > 4:
            exp = line.strip()
            indexes = []
            for w in exp.split()[:-1]:
                success = True
                if w not in vocab_list:
                    vocab_list.append(w)
                    try:
                        vecs.append(wv.word_to_vector(w.decode('utf8'))) 
                    except:
                        vecs.append(numpy.zeros(200,))
                        #print w
                        incomplete.append(w)
                indexes.append(vocab_list.index(w))

            destination_ok = True
            for w in exp.split()[:-1]:
                if w in incomplete:
                    #print '!', w
                    destination_ok = False

            if destination_ok:
                try:
                    if '.'.join([str(indexes[0]), str(indexes[1]), str(indexes[2])]) not in triplets:
                        examples.append((indexes, exp.split()[-1]))
                        print len(examples)
                        triplets.add('.'.join([str(indexes[0]), str(indexes[1]), str(indexes[2])]))
                    if len(examples) > limit:
                        break
                except:
                    print indexes

            else:
                corrupt_examples.append((indexes, exp.split()[-1]))

    return examples, vocab_list, vecs
    #print len(corrupt_examples)
    #import pdb;pdb.set_trace()


if __name__ == "__main__":
    get_data_with_negs()
