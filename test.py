import os
from options.test_options import TestOptions
from data.custom_dataset_data_loader import CustomDatasetDataLoader
from models.models import ModelsFactory
from util.tb_visualizer import TBVisualizer
import numpy as np
from tqdm import tqdm


class Test:
    def __init__(self):
        self._opt = TestOptions().parse()

        # load dataset
        data_loader_test = CustomDatasetDataLoader(self._opt, is_for_train=False)
        self._dataset_test = data_loader_test.load_data()
        self._dataset_test_size = len(data_loader_test)
        print('#test images = %d' % self._dataset_test_size)

        # load model
        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)

        # create visualizer
        self._tb_visualizer = TBVisualizer(self._opt)

        # test model
        self._prepare_directories()
        self._test_dataset()

    def _prepare_directories(self):
        self._output_dir = os.path.join(self._opt.output_estimation_dir, self._opt.name)
        if not os.path.isdir(self._output_dir):
            os.makedirs(self._output_dir)

    def _test_dataset(self):
        # set model to eval mode
        self._model.set_eval()

        for i_val_batch, val_batch in enumerate(tqdm(self._dataset_test)):
            # evaluate batch
            self._model.set_input(val_batch)
            self._model.forward(keep_estimation=True)

            # save estimation
            paths = self._model.get_current_paths()
            estims = self._model.get_last_saved_estimation()
            self._save_estimates(paths, estims)

    def _save_estimates(self, paths, estims):
        """
        :return: Estimates are stored as pos_bb<top,left,bottom,right>, pos_prob, neg_prob
        """
        num_samples = len(paths['pos_img'])
        for i in xrange(num_samples):

            # get estimation data
            estim_pos_bb_lowres = np.reshape(estims['estim_pos_bb_lowres'][i, ...], -1)
            estim_pos_prob = estims['estim_pos_prob'][i]
            estim_neg_prob = estims['estim_neg_prob'][i]
            estimation = np.concatenate([estim_pos_bb_lowres, estim_pos_prob, estim_neg_prob])

            # save estimation
            filename = os.path.splitext(os.path.basename(paths['pos_img'][i]))[0]
            output_path = os.path.join(self._output_dir, filename+'.csv')
            np.savetxt(output_path, estimation, delimiter=",")



if __name__ == "__main__":
    Test()