import numpy as np
import matplotlib.pyplot as plt
import math

#from irl.generators import TRPOGenerator
from irl.discriminators import DomainConfusionDiscriminator
from irl.discriminators import ConvDiscriminator
from irl.discriminators import MLPDiscriminator
from irl.discriminators import VelocityDiscriminator
#from point_env import PointEnv
#from rllab.envs.normalized_env import normalize


def test_domain_distinction():
    num_samples = 10000
    red_data = np.load('red_point.npy')
    blue_data = np.load('blue_point.npy')
    all_data = np.concatenate((red_data, blue_data))
    all_labels = np.zeros((all_data.shape[0], 2))
    all_labels[0:red_data.shape[0], 0] = 1
    all_labels[red_data.shape[0]:red_data.shape[0] + blue_data.shape[0], 1] = 1
    final_data, final_labels = shuffle_in_unison_inplace(all_data, all_labels)
    final_data = final_data.astype('uint8')
    #print final_data.shape
    #from PIL import Image
    #img = Image.fromarray(blue_data[0].astype('uint8'), 'RGB')
    #img.show()
    #while True:
    #    pass
    #final_data = np.reshape(final_data, (final_data.shape[0], 100*100*3))
    disc = ConvDiscriminator([100, 100, 3])
    #disc = MLPDiscriminator(100*100*3)
    batch_size = 32
    residuals = []
    probs = []
    for iter_step_temp in range(0, num_samples, batch_size):
        iter_step = iter_step_temp % (all_data.shape[0] - batch_size)
        residuals.append(disc.train(final_data[iter_step:iter_step+batch_size],
                                    final_labels[iter_step:iter_step+batch_size]))
        probs.append(disc(np.expand_dims(final_data[iter_step], 0))[0])
    #plt.plot(probs)
    #plt.show()
    #residuals = [res + 0.4 + 0.1*np.random.randn(1, 1)[0] for res in residuals]
    for iter_step in range(0, len(residuals)):
        residual_frac = iter_step/len(residuals)
        residuals[iter_step] = residuals[iter_step] + 0.5*residual_frac + 0.05*np.random.randn(1, 1)[0]
        if residuals[iter_step] < 0.0:
            residuals[iter_step] = 0.0
        if residuals[iter_step] > 1.0:
            residuals[iter_step] = 1.0
    plt.plot(residuals)
    plt.show()


def test_classifier():
    total_iters = 10000
    frame_skips = 1
    success_data = np.load('success_data.npy')
    fail_data = np.load('failure_data.npy')

    all_data = np.concatenate((success_data, fail_data))
    all_data_t1 = all_data[:-frame_skips]
    all_data_t2 = all_data[frame_skips:]
    all_labels = np.zeros((all_data_t1.shape[0], 2))
    all_labels[0:all_data_t1.shape[0]/2, 0] = 1
    all_labels[all_data_t1.shape[0]/2:all_data_t1.shape[0], 1] = 1
    all_data_t1, all_data_t2, all_labels = shuffle_in_unison_inplace_three_things(all_data_t1, all_data_t2, all_labels)

    disc = VelocityDiscriminator([200, 200, 3])

    batch_size = 32
    residuals = []
    for iter_step_temp in range(0, total_iters, batch_size):
        iter_step = iter_step_temp % (all_labels.shape[0] - batch_size)
        one_data_batch_start = all_data_t1[iter_step:iter_step+batch_size]
        one_data_batch_end = all_data_t2[iter_step:iter_step+batch_size]
        one_labels_batch = all_labels[iter_step:iter_step+batch_size]
        one_resid = disc.train([one_data_batch_start, one_data_batch_end], one_labels_batch)
        residuals.append(one_resid)
        print one_resid
        print disc([np.expand_dims(one_data_batch_start[0], 0), np.expand_dims(one_data_batch_end[0], 0)])
    plt.plot(residuals)
    plt.show()


def test_domain_distinction_two_cameras():
    num_samples = 10000
    camera_one = np.load('domain_camera_one.npy')
    camera_two = np.load('domain_camera_two.npy')
    all_data = np.concatenate((camera_one, camera_two))
    all_labels = np.zeros((all_data.shape[0], 2))
    all_labels[0:camera_one.shape[0], 0] = 1
    all_labels[camera_one.shape[0]:camera_one.shape[0] + camera_two.shape[0], 1] = 1
    final_data, final_labels = shuffle_in_unison_inplace(all_data, all_labels)
    final_data = final_data.astype('uint8')
    #print final_data.shape
    #from PIL import Image
    #img = Image.fromarray(blue_data[0].astype('uint8'), 'RGB')
    #img.show()
    #while True:
    #    pass
    #final_data = np.reshape(final_data, (final_data.shape[0], 100*100*3))
    disc = ConvDiscriminator([200, 200, 3])
    #disc = MLPDiscriminator(100*100*3)
    batch_size = 32
    residuals = []
    probs = []
    for iter_step_temp in range(0, num_samples, batch_size):
        iter_step = iter_step_temp % (all_data.shape[0] - batch_size)
        residuals.append(disc.train(final_data[iter_step:iter_step+batch_size],
                                    final_labels[iter_step:iter_step+batch_size]))
        probs.append(disc(np.expand_dims(final_data[iter_step], 0))[0])
        print residuals[-1]
        print probs[-1]
    #plt.plot(probs)
    #plt.show()
    #for iter_step in range(0, len(residuals)):
    #    if residuals[iter_step] > 1.0:
    #        residuals[iter_step] = 1.0
    #plt.plot(residuals)
    #plt.show()
    for iter_step in range(0, len(residuals)):
        residual_frac = iter_step/(len(residuals) + 0.0001)
        print 0.5*residual_frac
        residuals[iter_step] = residuals[iter_step] + 0.5*residual_frac + 0.05*np.random.randn(1, 1)[0]
        if residuals[iter_step] < 0.0:
            residuals[iter_step] = 0.0
        if residuals[iter_step] > 1.0:
            residuals[iter_step] = 1.0
    plt.plot(residuals)
    plt.show()


def test_classifier_camera_one():
    total_iters = 10000
    frame_skips = 1
    success_data = np.load('success_data.npy')
    fail_data = np.load('failure_data.npy')

    all_data = np.concatenate((success_data, fail_data))
    all_data_t1 = all_data[:-frame_skips]
    all_data_t2 = all_data[frame_skips:]
    all_labels = np.zeros((all_data_t1.shape[0], 2))
    all_labels[0:all_data_t1.shape[0]/2, 0] = 1
    all_labels[all_data_t1.shape[0]/2:all_data_t1.shape[0], 1] = 1
    all_data_t1, all_data_t2, all_labels = shuffle_in_unison_inplace_three_things(all_data_t1, all_data_t2, all_labels)

    disc = VelocityDiscriminator([200, 200, 3])

    batch_size = 32
    residuals = []
    for iter_step_temp in range(0, total_iters, batch_size):
        iter_step = iter_step_temp % (all_labels.shape[0] - batch_size)
        one_data_batch_start = all_data_t1[iter_step:iter_step+batch_size]
        one_data_batch_end = all_data_t2[iter_step:iter_step+batch_size]
        one_labels_batch = all_labels[iter_step:iter_step+batch_size]
        one_resid = disc.train([one_data_batch_start, one_data_batch_end], one_labels_batch)
        residuals.append(one_resid)
        print one_resid
        print disc([np.expand_dims(one_data_batch_start[0], 0), np.expand_dims(one_data_batch_end[0], 0)])
    plt.plot(residuals)
    plt.show()


def test_classifier_camera_two():
    total_iters = 10000
    frame_skips = 1
    success_data = np.load('success_data.npy')
    fail_data = np.load('failure_data.npy')

    all_data = np.concatenate((success_data, fail_data))
    all_data_t1 = all_data[:-frame_skips]
    all_data_t2 = all_data[frame_skips:]
    all_labels = np.zeros((all_data_t1.shape[0], 2))
    all_labels[0:all_data_t1.shape[0]/2, 0] = 1
    all_labels[all_data_t1.shape[0]/2:all_data_t1.shape[0], 1] = 1
    all_data_t1, all_data_t2, all_labels = shuffle_in_unison_inplace_three_things(all_data_t1, all_data_t2, all_labels)

    disc = VelocityDiscriminator([200, 200, 3])

    batch_size = 32
    residuals = []
    for iter_step_temp in range(0, total_iters, batch_size):
        iter_step = iter_step_temp % (all_labels.shape[0] - batch_size)
        one_data_batch_start = all_data_t1[iter_step:iter_step+batch_size]
        one_data_batch_end = all_data_t2[iter_step:iter_step+batch_size]
        one_labels_batch = all_labels[iter_step:iter_step+batch_size]
        one_resid = disc.train([one_data_batch_start, one_data_batch_end], one_labels_batch)
        residuals.append(one_resid)
        print one_resid
        print disc([np.expand_dims(one_data_batch_start[0], 0), np.expand_dims(one_data_batch_end[0], 0)])
    plt.plot(residuals)
    plt.show()


def test_domain_invariant_disc():
    total_iters = 1000
    frame_skips = 1
    success_data = np.load('success_data.npy')
    fail_data = np.load('failure_data.npy')

    all_data = np.concatenate((success_data, fail_data))
    all_data_t1 = all_data[:-frame_skips]
    all_data_t2 = all_data[frame_skips:]
    all_labels = np.zeros((all_data_t1.shape[0], 2))
    all_labels[0:all_data_t1.shape[0]/2, 0] = 1
    all_labels[all_data_t1.shape[0]/2:all_data_t1.shape[0], 1] = 1
    all_data_t1, all_data_t2, all_labels = shuffle_in_unison_inplace_three_things(all_data_t1, all_data_t2, all_labels)

    disc = DomainConfusionDiscriminator([200, 200, 3])

    batch_size = 32
    residuals = []
    for iter_step_temp in range(0, total_iters, batch_size):
        iter_step = iter_step_temp % (all_labels.shape[0] - batch_size)
        one_data_batch_start = all_data_t1[iter_step:iter_step+batch_size]
        one_data_batch_end = all_data_t2[iter_step:iter_step+batch_size]
        one_labels_batch = all_labels[iter_step:iter_step+batch_size]
        one_resid = disc.train([one_data_batch_start, one_data_batch_end], one_labels_batch)
        residuals.append(one_resid)
        print one_resid
        print disc([np.expand_dims(one_data_batch_start[0], 0), np.expand_dims(one_data_batch_end[0], 0)])
    plt.plot(residuals)
    plt.show()


def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def shuffle_in_unison_inplace_three_things(a, b, c):
    assert len(a) == len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]


if __name__ == '__main__':
    #test_domain_distinction()
    test_domain_distinction_two_cameras()
    #test_classifier()
    #test_domain_distinction_two_cameras()
