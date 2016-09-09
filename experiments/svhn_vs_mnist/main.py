import pickle as pkl
import numpy as np
import irl.discriminators as discrims

from tensorflow.examples.tutorials.mnist import input_data

def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Process MNIST
    mnist_train = (mnist.train.images > 0).reshape(55000, 28, 28, 1).astype(np.uint8) * 255
    mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
    mnist_test = (mnist.test.images > 0).reshape(10000, 28, 28, 1).astype(np.uint8) * 255
    mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)

    # Load MNIST-M
    mnistm = pkl.load(open('mnistm_data.pkl'))
    mnistm_train = mnistm['train']
    mnistm_test = mnistm['test']
    mnistm_valid = mnistm['valid']

    # Compute pixel mean for normalizing data
    pixel_mean = np.vstack([mnist_train, mnistm_train]).mean((0, 1, 2))

    # Create a mixed dataset for TSNE visualization
    mnist_test = mnist_train
    mnistm_test = mnistm_train
    num_test = mnist_test.shape[0]
    combined_test_imgs = np.vstack([mnist_test[:num_test], mnistm_test[:num_test]])
    combined_test_labels = np.vstack([mnist.train.labels[:num_test], mnist.train.labels[:num_test]])
    combined_test_domain = np.vstack([np.tile([1., 0.], [num_test, 1]),
                                      np.tile([0., 1.], [num_test, 1])])

    combined_test_imgs, combined_test_labels, combined_test_domain = shuffle_in_unison_inplace_three_things(combined_test_imgs,
                                                                                                            combined_test_labels,
                                                                                                            combined_test_domain)

    batcher = Batcher(combined_test_imgs, combined_test_labels, combined_test_domain,
                      batch_size=128, total_data_size=combined_test_imgs.shape[0])

    disc = discrims.DomainConfusionDiscriminator(input_dim=combined_test_imgs.shape[1:4],
                                                 output_dim_class=combined_test_labels.shape[1],
                                                 output_dim_dom=combined_test_domain.shape[1])

    epoch = 0
    prev_epoch = 0
    tot_loss = 0
    all_losses = []
    mega_all_losses = []
    while epoch < 3:
        epoch = batcher.epoch
        if epoch > prev_epoch:
            print (tot_loss / batcher.total_batches)
            print 'starting new epoch'
            all_losses.append(tot_loss / batcher.total_batches)
            tot_loss = 0
        data_batch, lab_batch, dom_batch = batcher()
        disc.train(data_batch, [lab_batch, dom_batch])
        one_dom_loss = disc.get_dom_accuracy(data_batch, dom_batch)
        one_lab_loss = disc.get_lab_accuracy(data_batch, lab_batch)
        #one_loss = one_lab_loss
        #tot_loss += one_loss
        #all_losses.append(one_loss)
        #mega_all_losses.append(one_loss)
        print one_dom_loss
        mega_all_losses.append(one_dom_loss)
        #print one_loss
        prev_epoch = epoch
    import matplotlib.pyplot as plt
    plt.plot(mega_all_losses)
    plt.show()
    plt.plot(all_losses)


class Batcher:
    def __init__(self, data, labels, domains, batch_size, total_data_size):
        self.data = data
        self.labels = labels
        self.domains = domains
        self.batch_size = batch_size
        self.total_data_size = total_data_size
        self.last_batch_idx = 0
        self.total_batches = self.total_data_size / self.batch_size
        self.epoch = -1

    def __call__(self):
        self.last_batch_idx = self.last_batch_idx % self.total_batches
        start = self.last_batch_idx*self.batch_size
        stop = (self.last_batch_idx+1)*self.batch_size
        dat = self.data[start:stop]
        lab = self.labels[start:stop]
        dom = self.domains[start:stop]
        if self.last_batch_idx == 0:
            self.epoch += 1
        self.last_batch_idx += 1
        return dat, lab, dom


def shuffle_in_unison_inplace_three_things(a, b, c):
    print len(a)
    print len(b)
    assert len(a) == len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]



main()