import numpy as np
from scipy.special import softmax

def get_impartial_outputs(out, classification_tasks, mean, std):
    output = {}

    if len(out.shape) <= 4: #there are no multiple predictions as in MCdropout or Ensemble: dims are batchxchannelsxwxh
        ix = 0
        for class_tasks_key in classification_tasks.keys():
            output_task = {}

            classification_tasks = classification_tasks[class_tasks_key]
            nclasses = int(classification_tasks['classes'])  # number of classes
            rec_channels = classification_tasks['rec_channels']  # list with channels to reconstruct
            nrec_channels = len(rec_channels)

            ncomponents = np.array(classification_tasks['ncomponents'])
            out_seg = softmax(out[:, ix:ix + np.sum(ncomponents), ...], axis = 1)
            ix += np.sum(ncomponents)

            ## class segmentations
            out_classification = np.zeros([out_seg.shape[0], nclasses, out_seg.shape[2], out_seg.shape[3]])

            ix_seg = 0
            for ix_class in range(nclasses):
                out_classification[:,ix_class,...] = np.sum(out_seg[:, ix_seg:ix_seg + ncomponents[ix_class], ...], 1)
                ix_seg += ncomponents[ix_class]
            output_task['class_segmentation'] = out_classification

            ### Factors & Reconstruction Loss ###
            output_factors = {}
            output_factors['components'] = out_seg
            if nrec_channels > 0:
                for ch in rec_channels:
                    if mean:
                        output_factors['mean_ch' + str(ch)] = out[:, ix:ix + np.sum(ncomponents), ...]
                        ix += np.sum(ncomponents)

                    if std:
                        output_factors['std_ch' + str(ch)] = out[:, ix:ix + np.sum(ncomponents), ...]
                        ix += np.sum(ncomponents)
            output_task['factors'] = output_factors

            #task
            output[class_tasks_key] = output_task
    else:
        ix = 0
        # epsilon = sys.float_info.min
        for class_tasks_key in classification_tasks.keys():
            output_task = {}

            classification_tasks = classification_tasks[class_tasks_key]
            nclasses = int(classification_tasks['classes'])  # number of classes
            rec_channels = classification_tasks['rec_channels']  # list with channels to reconstruct
            nrec_channels = len(rec_channels)
            ncomponents = np.array(classification_tasks['ncomponents'])

            out_seg = softmax(out[:, :, ix:ix + np.sum(ncomponents), ...], axis=2)  #size : predictions, batch, channels , h, w
            ix += np.sum(ncomponents)

            ## class segmentations
            mean_classification = np.zeros([out_seg.shape[1],nclasses,out_seg.shape[-2],out_seg.shape[-1]])
            variance_classification = np.zeros([out_seg.shape[1], nclasses, out_seg.shape[-2], out_seg.shape[-1]])

            ix_seg = 0
            for ix_class in range(nclasses):
                aux = np.sum(out_seg[:,:, ix_seg:ix_seg + ncomponents[ix_class], ...], 2) #size : predictions, batch, h, w
                mean_classification[:,ix_class,...] = np.mean(aux,axis=0) #batch, h, w
                variance_classification[:,ix_class,...] = np.var(aux,axis=0)
                ix_seg += ncomponents[ix_class]
            output_task['class_segmentation'] = mean_classification
            output_task['class_segmentation_variance'] = variance_classification

            ### Factors & Reconstruction Loss ###
            output_factors = {}
            output_factors['components'] = np.mean(out_seg, axis=0)
            output_factors['components_variance'] = np.var(out_seg, axis=0)

            if nrec_channels > 0:
                for ch in rec_channels:
                    if mean:
                        output_factors['mean_ch' + str(ch)] = np.mean(out[:, :, ix:ix + np.sum(ncomponents), ...], axis=0)
                        output_factors['mean_variance_ch' + str(ch)] = np.var(out[:, :, ix:ix + np.sum(ncomponents), ...], axis=0)
                        ix += np.sum(ncomponents)

                    if std:
                        output_factors['logstd_ch' + str(ch)] = np.mean(out[:, :, ix:ix + np.sum(ncomponents), ...], axis=0)
                        output_factors['logstd_variance_ch' + str(ch)] = np.var(out[:, :, ix:ix + np.sum(ncomponents), ...], axis=0)
                        ix += np.sum(ncomponents)
            output_task['factors'] = output_factors

            #task
            output[class_tasks_key] = output_task

    return output




# # TODO: Delete this 
# def get_impartial_outputs(out, config):
#     output = {}

#     if len(out.shape) <= 4: #there are no multiple predictions as in MCdropout or Ensemble: dims are batchxchannelsxwxh
#         ix = 0
#         for class_tasks_key in config.classification_tasks.keys():
#             output_task = {}

#             classification_tasks = config.classification_tasks[class_tasks_key]
#             nclasses = int(classification_tasks['classes'])  # number of classes
#             rec_channels = classification_tasks['rec_channels']  # list with channels to reconstruct
#             nrec_channels = len(rec_channels)

#             ncomponents = np.array(classification_tasks['ncomponents'])
#             out_seg = softmax(out[:, ix:ix + np.sum(ncomponents), ...], axis = 1)
#             ix += np.sum(ncomponents)

#             ## class segmentations
#             out_classification = np.zeros([out_seg.shape[0], nclasses, out_seg.shape[2], out_seg.shape[3]])

#             ix_seg = 0
#             for ix_class in range(nclasses):
#                 out_classification[:,ix_class,...] = np.sum(out_seg[:, ix_seg:ix_seg + ncomponents[ix_class], ...], 1)
#                 ix_seg += ncomponents[ix_class]
#             output_task['class_segmentation'] = out_classification

#             ### Factors & Reconstruction Loss ###
#             output_factors = {}
#             output_factors['components'] = out_seg
#             if nrec_channels > 0:
#                 for ch in rec_channels:
#                     if config.mean:
#                         output_factors['mean_ch' + str(ch)] = out[:, ix:ix + np.sum(ncomponents), ...]
#                         ix += np.sum(ncomponents)

#                     if config.std:
#                         output_factors['std_ch' + str(ch)] = out[:, ix:ix + np.sum(ncomponents), ...]
#                         ix += np.sum(ncomponents)
#             output_task['factors'] = output_factors

#             #task
#             output[class_tasks_key] = output_task
#     else:
#         ix = 0
#         # epsilon = sys.float_info.min
#         for class_tasks_key in config.classification_tasks.keys():
#             output_task = {}

#             classification_tasks = config.classification_tasks[class_tasks_key]
#             nclasses = int(classification_tasks['classes'])  # number of classes
#             rec_channels = classification_tasks['rec_channels']  # list with channels to reconstruct
#             nrec_channels = len(rec_channels)
#             ncomponents = np.array(classification_tasks['ncomponents'])

#             out_seg = softmax(out[:, :, ix:ix + np.sum(ncomponents), ...], axis=2)  #size : predictions, batch, channels , h, w
#             ix += np.sum(ncomponents)

#             ## class segmentations
#             mean_classification = np.zeros([out_seg.shape[1],nclasses,out_seg.shape[-2],out_seg.shape[-1]])
#             variance_classification = np.zeros([out_seg.shape[1], nclasses, out_seg.shape[-2], out_seg.shape[-1]])

#             ix_seg = 0
#             for ix_class in range(nclasses):
#                 aux = np.sum(out_seg[:,:, ix_seg:ix_seg + ncomponents[ix_class], ...], 2) #size : predictions, batch, h, w
#                 mean_classification[:,ix_class,...] = np.mean(aux,axis=0) #batch, h, w
#                 variance_classification[:,ix_class,...] = np.var(aux,axis=0)
#                 ix_seg += ncomponents[ix_class]
#             output_task['class_segmentation'] = mean_classification
#             output_task['class_segmentation_variance'] = variance_classification

#             ### Factors & Reconstruction Loss ###
#             output_factors = {}
#             output_factors['components'] = np.mean(out_seg, axis=0)
#             output_factors['components_variance'] = np.var(out_seg, axis=0)

#             if nrec_channels > 0:
#                 for ch in rec_channels:
#                     if config.mean:
#                         output_factors['mean_ch' + str(ch)] = np.mean(out[:, :, ix:ix + np.sum(ncomponents), ...], axis=0)
#                         output_factors['mean_variance_ch' + str(ch)] = np.var(out[:, :, ix:ix + np.sum(ncomponents), ...], axis=0)
#                         ix += np.sum(ncomponents)

#                     if config.std:
#                         output_factors['logstd_ch' + str(ch)] = np.mean(out[:, :, ix:ix + np.sum(ncomponents), ...], axis=0)
#                         output_factors['logstd_variance_ch' + str(ch)] = np.var(out[:, :, ix:ix + np.sum(ncomponents), ...], axis=0)
#                         ix += np.sum(ncomponents)
#             output_task['factors'] = output_factors

#             #task
#             output[class_tasks_key] = output_task

#     return output
