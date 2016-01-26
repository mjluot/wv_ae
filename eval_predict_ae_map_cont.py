import theano
import theano.tensor as T
from theano import config
import random
import numpy
from wvlib import wvlib
import pickle
from scipy.spatial.distance import cosine

class simple_ae_layer():

    def __init__(self, input, n_in, n_out, n_hidden, rng):

        self.input = input

        #Make W and b
        WV = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_hidden)),
                    high=numpy.sqrt(6. / (n_in + n_hidden)),
                    size=(n_in, n_hidden)
                ),
                dtype=theano.config.floatX
            )
        self.W = theano.shared(value=WV, name='W', borrow=True)
        b_values = numpy.zeros((n_hidden,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        #self.hidden_output = T.tanh(T.dot(self.input, self.W) + self.b)

        self.hidden_output = T.dot(self.input, self.W)


        WV2 = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_hidden)),
                    high=numpy.sqrt(6. / (n_in + n_hidden)),
                    size=(n_hidden, n_out)
                ),
                dtype=theano.config.floatX
            )
        self.W2 = theano.shared(value=WV2, name='W', borrow=True)
        b_values2 = numpy.zeros((n_out,), dtype=theano.config.floatX)
        self.b2 = theano.shared(value=b_values2, name='b', borrow=True)

        #self.output = T.tanh(T.dot(self.hidden_output, self.W2) + self.b2)
        self.output = T.dot(self.hidden_output, self.W2)
        self.params = [self.W, self.W2]

        
#class simple_concat_layer
#class simple_split_layer
#


class linear_mapping_layer():

    def __init__(self, input, n_in, n_out, rng):

        self.input = input
        WV = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

        self.W = theano.shared(value=WV, name='W', borrow=True)
        self.output = T.dot(self.input, self.W)
        self.rev_output = T.dot(self.input, self.W.T)
        self.params = [self.W]


class concatenation_layer():

    def __init__(self, input):

        self.input = input
        self.output = T.concatenate(self.input, axis=1)

class split_layer():

     def __init__(self, input, n_vecs, vec_size):
         self.input = input
         self.output = []
         for i in range(n_vecs):
             self.output.append(input[:,i*vec_size:(i+1)*vec_size])

def main():

    '''
    made_up_data = []
    for i in range(100):
        made_up_data.append([random.randint(1,6) / 10.0 for y in range(3)] + [random.randint(6,9) / 10.0 for y in range(3)])

    #Let's make a simple ae
    split_data = numpy.array(made_up_data)[:,:3], numpy.array(made_up_data)[:,3:]
    #Let's test the concatenation layer thing!
    '''
    #Config stuff
    n_in_vecs = 2
    vec_size = 200
    hidden_size = 200
    minibatch_size = 1000

    #Let us load the data
    inf = open('dataset_xm.list','rb')
    examples, vocab, vecs = pickle.load(inf)#)vector_grabber.get_data()
    inf.close()
    #examples, vocab, vecs = vector_grabber_mini.get_data()
    print len(examples)
    #import pdb;pdb.set_trace()
    #out = open('the_clean_data_mini','wb')
    #pickle.dump((examples, vocab, vecs), out)
    #out.close()
    #import pdb;pdb.set_trace()
    #vec table made!

    #import pdb;pdb.set_trace()

    n_vecs = []
    for v in vecs:
        if len(v) < 300:
            n_vecs.append(v)
        else:
            n_vecs.append(numpy.zeros(200))

    vec_table = theano.shared(value=numpy.array(n_vecs), name='W', borrow=True)

    n_examples = []
    for e in examples:
        if e[-1].startswith('X'):
            n_examples.append(e[:-1])

    eval_cong = []
    eval_incong = []
    for e in examples:
        if e[-1].startswith('c'):
            eval_cong.append(e[:-1])
        elif e[-1].startswith('i'):
            eval_incong.append(e[:-1])

    #minibatches
    minibatches = []
    for i in range(0,len(n_examples), minibatch_size):
        minibatches.append(numpy.array(n_examples[i:i+minibatch_size]))

    #Let's create input variables:
    input_variables = [T.lvector() for i in range(n_in_vecs)]
    inputs = [vec_table[input_variables[0]], vec_table[input_variables[1]]]
    conc_layer = concatenation_layer(inputs)

    o_input = T.lvector()
    rng = numpy.random.RandomState(1234)

    #SV_ae
    sv_ae = simple_ae_layer(conc_layer.output, vec_size*n_in_vecs, vec_size*n_in_vecs, hidden_size, rng)
    sv_sp_layer = split_layer(sv_ae.output, n_in_vecs, vec_size)
    sv_res_f = theano.function(input_variables, sv_sp_layer.output)
    sv_cost = T.mean((conc_layer.output - sv_ae.output) ** 2)
    sv_cost_f = theano.function(input_variables, [sv_cost])

    learning_rate = theano.shared(0.8)

    sv_gparams = [T.grad(sv_cost, param) for param in sv_ae.params]
    updates = [(param, param - learning_rate * gparam) for param, gparam in zip(sv_ae.params, sv_gparams)]
    sv_train_f = theano.function(input_variables, sv_cost, updates=updates)

    #Mapping
    mapl = linear_mapping_layer(sv_ae.hidden_output, vec_size, hidden_size, rng)
    m_cost = T.mean((vec_table[o_input] - mapl.output) ** 2)
    #learning_rate = theano.shared(0.5)
    gparams = [T.grad(m_cost, param) for param in mapl.params]
    das_parameters = [mapl.W, sv_ae.W]
    updates = [(param, param - learning_rate * gparam) for param, gparam in zip(das_parameters, gparams)]
    m_train_f = theano.function([input_variables[0], input_variables[1], o_input], m_cost, updates=updates)
    m_cost_f = theano.function([input_variables[0], input_variables[1], o_input], m_cost)
    #
    res_f = theano.function(inputs, mapl.output)
    vec_f = theano.function([o_input], vec_table[o_input])
    gres_f = theano.function(input_variables, mapl.output)
    vres_f = theano.function(inputs, sv_ae.hidden_output)

    #Now let us do evaluation related stuffs!
    #Let us load up a model
    #model__latest_200_ae_map680
    inf = open('./models_ae/model__latest_200_ae_map680','rb')#model_200_ae_map7880','rb')#'./models_ae/model_200_ae_map620','rb')
    params = pickle.load(inf)
    #pickle.dump([sv_ae.params, mapl.params], outf)
    for p,a in zip(params[0], sv_ae.params):
        a.set_value(p.get_value())
    for p,a in zip(params[1], mapl.params):
        a.set_value(p.get_value())
    inf.close()

    xc = len(eval_cong)
    c_results = gres_f(numpy.array(eval_cong).reshape(xc,3)[:,0], numpy.array(eval_cong).reshape(xc,3)[:,1])

    xi = len(eval_incong)
    ic_results = gres_f(numpy.array(eval_incong).reshape(xi,3)[:,0], numpy.array(eval_incong).reshape(xi,3)[:,1])

    c_objs = vec_f(numpy.array(eval_cong).reshape(xc,3)[:,2])
    ic_objs = vec_f(numpy.array(eval_incong).reshape(xi,3)[:,2])

    #Find the pairs
    the_pairs_c = []
    the_pairs_ic = []
    for trp in eval_cong:
        the_pairs_c.append(str(trp[0][0]) + '+' + str(trp[0][1]))
    for trp in eval_incong:
        the_pairs_ic.append(str(trp[0][0]) + '+' + str(trp[0][1]))

    pair_map = []
    for trp in eval_cong:
        if str(trp[0][0]) + '+' + str(trp[0][1]) in the_pairs_c and str(trp[0][0]) + '+' + str(trp[0][1]) in the_pairs_ic:
            pair_map.append((the_pairs_c.index(str(trp[0][0]) + '+' + str(trp[0][1])),  the_pairs_ic.index(str(trp[0][0]) + '+' + str(trp[0][1]))))

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    count = 0

    for tpl in pair_map:
        cd = cosine(c_objs[tpl[0]], c_results[tpl[0]])
        icd = cosine(ic_objs[tpl[1]], ic_results[tpl[1]])
        print cd, icd
        if cd < icd:
            count += 1

    print count/float(len(pair_map))



    #The fun loop
    #wv = wvlib.load("/usr/share/ParseBank/vector-space-models/FIN/w2v_pbv3_lm.rev01.bin",max_rank=100000)
    #import pdb;pdb.set_trace()
    wv = wvlib.load("/home/ginter/w2v/pb34_lemma_200_v2.bin",max_rank=50000).normalize()
    import pdb;pdb.set_trace()
    #The loop
    while True:
        try:
            xs = raw_input('Subject ')
            xv = raw_input('Verb ')
            xsv = wv.word_to_vector(xs.decode('utf8'))
            xvv = wv.word_to_vector(xv.decode('utf8'))

            if xs.startswith('exit') or xv.startswith('exit'):
                break

            for t in wv.approximate_nearest(res_f([xsv], [xvv])[0]):#wv.nearest(vres_f(numpy.array([xsv,]),numpy.array([xvv]))[0]):
                print t
            print
        except:
            pass



    '''
    for mb in range(100000):


        if mb%10 == 0:
            r_examples = n_examples[:]
            random.shuffle(r_examples)
            minibatches = []
            for i in range(0,len(r_examples), minibatch_size):
                minibatches.append(numpy.array(r_examples[i:i+minibatch_size]))

        sve_cost = []
        me_cost = []

        for b in range(0,len(minibatches)):
            try:

                if mb > 20:
                    svb_cost = sv_train_f(minibatches[b].reshape(minibatch_size,3)[:,0], minibatches[b].reshape(minibatch_size,3)[:,1])
                    sve_cost.append(svb_cost)
  
                mb_cost = m_train_f(minibatches[b].reshape(minibatch_size,3)[:,0], minibatches[b].reshape(minibatch_size,3)[:,1], minibatches[b].reshape(minibatch_size,3)[:,2])

                #print svb_cost, sob_cost, ovb_cost, b
                me_cost.append(mb_cost)
            except:
                pass

        #print numpy.mean(e_cost)
        if mb%20 == 0:
            #Decay
            learning_rate = learning_rate * 0.9

            outf = open('./models_ae/model_200_ae_map' + str(mb), 'wb')
            pickle.dump([sv_ae.params, mapl.params], outf)
            outf.close()

        if True:#mb%20 == 0:
            print
            print numpy.mean(me_cost), mb
            #Evaluate!
            print '<--eval-->'
            x = len(eval_cong)
            print 'Congruent'
            print 'mapping', m_cost_f(numpy.array(eval_cong).reshape(x,3)[:,0], numpy.array(eval_cong).reshape(x,3)[:,1], numpy.array(eval_cong).reshape(x,3)[:,2])

            x = len(eval_incong)
            print 'Incongruent'
            print 'mapping', m_cost_f(numpy.array(eval_incong).reshape(x,3)[:,0], numpy.array(eval_incong).reshape(x,3)[:,1], numpy.array(eval_incong).reshape(x,3)[:,2])
            print '</--eval-->'
            #import pdb;pdb.set_trace()

    import pdb;pdb.set_trace()
    '''

 
    

    



main()

