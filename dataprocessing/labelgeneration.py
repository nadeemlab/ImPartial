import numpy as np
from skimage import morphology
# from preprocessing import generate_patches_syxc
# from csbdeep.data import RawData
import matplotlib.pyplot as plt

def get_scribbles_mask(label_image, fov_box=(32, 32), max_labels=4,
                       radius_pointer=0, disk_scribble=False,
                       sample_back = False, previous_scribbles = None):
    mask_image = np.zeros_like(label_image)
    mask_image[label_image > 0] = 1.0

    nlabels = np.unique(label_image[label_image > 0]).shape[0]
    max_labels = np.minimum(max_labels, nlabels)

    exclude_labels = []
    if previous_scribbles is not None:
        exclude_labels = list(np.unique(label_image[(label_image > 0) & (previous_scribbles[...,0]>0)]))
        max_labels = np.minimum(max_labels, nlabels - len(exclude_labels))




    ### Set the instance scribble mask mask_sk
    mask_sk = morphology.skeletonize(mask_image) * mask_image
    if radius_pointer > 0:
        selem = morphology.disk(radius_pointer)
        mask_sk = morphology.dilation(mask_sk, selem=selem) * mask_image
    if disk_scribble:
        selem = morphology.disk(3)
        outline_arc = morphology.erosion(mask_image, selem=selem) - \
                      morphology.erosion(morphology.erosion(mask_image, selem=selem), selem=morphology.disk(1))
        if radius_pointer > 0:
            selem = morphology.disk(radius_pointer)
            outline_arc = morphology.dilation(outline_arc, selem=selem)

        mask_sk += outline_arc
        mask_sk[mask_sk > 0] = 1
        mask_sk = mask_sk * mask_image

    nbudget = max_labels + 0
    labels_image_res = np.array(label_image)

    back_aux = np.array(mask_image)
    back_aux = morphology.dilation(back_aux, selem=morphology.disk(4))

    fov_image_res = np.array(1-mask_image)*back_aux #background entre foreground
    fov_image_res = morphology.skeletonize(fov_image_res)

    # plt.figure(figsize=(10,5))
    # plt.subplot(1,2,1)
    # plt.imshow(fov_image_res)
    # plt.subplot(1, 2, 2)

    back_aux = morphology.skeletonize(1-back_aux)
    fov_image_res += morphology.dilation(back_aux,selem=morphology.disk(4))

    # plt.imshow(fov_image_res)
    # plt.show()

    foreground = np.zeros_like(labels_image_res)
    background = np.zeros_like(labels_image_res)

    ## remove from labels_image_res labels to exclude:
    if len(exclude_labels) > 0:

        label_aux = np.zeros_like(labels_image_res)
        for label in exclude_labels:
            label_aux[labels_image_res == label] = 1

        labels_image_res = labels_image_res * (1 - label_aux)
        fov_image_res = fov_image_res * (1 - label_aux) #dont know if this is necessary
    #
    # plt.figure(figsize=(15, 5))
    # plt.subplot(1,3,1)
    # plt.imshow(fov_image_res)
    # plt.subplot(1, 3, 2)
    # plt.imshow(labels_image_res)
    # plt.subplot(1, 3, 3)
    # plt.imshow(label_image)
    # plt.show()

    while nbudget > 0:

        ## Pick a point at random
        if sample_back:
            aux = np.array(fov_image_res)
        else:
            aux = np.array(labels_image_res)

        ix0_back, ix1_back = np.nonzero(aux)
        ix = np.random.randint(0, ix0_back.shape[0])

        # bounding box
        xmin = np.maximum(0, ix0_back[ix] - int(fov_box[0] / 2))
        xmax = np.minimum(xmin + int(fov_box[0]), labels_image_res.shape[0])

        ymin = np.maximum(0, ix1_back[ix] - int(fov_box[1] / 2))
        ymax = np.minimum(ymin + int(fov_box[1]), labels_image_res.shape[0])

        bb_image = np.zeros_like(labels_image_res)
        bb_image[xmin:xmax, ymin:ymax] = 1

        active_labels = labels_image_res * bb_image

        if np.sum(active_labels>0)>0:
            active_labels = np.unique(active_labels[active_labels > 0])

            for label in active_labels:
                label_aux = np.zeros_like(labels_image_res)
                label_aux[labels_image_res == label] = 1

                ## foreground
                foreground += label_aux * mask_sk

                back_aux = morphology.dilation(label_aux, selem=morphology.disk(4))
                back_aux += bb_image
                back_aux = back_aux * (1 - mask_image)
                back_aux[back_aux > 0] = 1
                back_aux = morphology.skeletonize(back_aux)

                if radius_pointer > 0:
                    selem = morphology.disk(radius_pointer)
                    back_aux = morphology.dilation(back_aux, selem=selem) * (1 - mask_image)

                background += back_aux * (1 - mask_image)  # double secure

                nbudget -= 1
                labels_image_res = labels_image_res * (1 - label_aux)
                fov_image_res = fov_image_res * (1 - label_aux)
        else:
            # print('NO labels')
            if (np.sum(fov_image_res*bb_image)>0):
                back_aux = morphology.skeletonize(bb_image*(1-mask_image)*fov_image_res) ## box with background
                if radius_pointer > 0:
                    selem = morphology.disk(radius_pointer)
                    back_aux = morphology.dilation(back_aux, selem=selem) * (1 - mask_image)
                background += back_aux * (1 - mask_image)  # double secure
                # ix0_nonz, ix1_nonz = np.nonzero(back_aux)
                # # if radius_pointer > 0:
                # w=15
                # back_aux = np.zeros_like(back_aux)
                # back_aux[ix0_nonz[0],ix1_nonz[0]-w:ix1_nonz[0]+w] = 1
                # # selem[:,int(fov_box[1] / 2)] = 1
                #
                # # back_aux = morphology.dilation(back_aux, selem=selem) * (1 - mask_image)

        fov_image_res = fov_image_res * (1 - bb_image)

        foreground[foreground>0] = 1
        background[background>0] = 1

        print('budget : ', nbudget)

    mask_scribbles = np.zeros([mask_image.shape[0], mask_image.shape[1], 2])
    mask_scribbles[..., 0] = np.array(foreground)  # fore
    mask_scribbles[..., 1] = np.array(background)   # back

    if previous_scribbles is not None:
        print('Merging with previous scribbles')
        mask_scribbles = mask_scribbles + previous_scribbles
        mask_scribbles[mask_scribbles>0] = 1


    nscribbles = np.unique(label_image[(label_image > 0) & (mask_scribbles[...,0]>0)]).shape[0] - len(exclude_labels)



    return mask_scribbles, nscribbles, nlabels


def get_scribbles(train_masks, n_labels_total, fov_box=(32, 32),
                  radius_pointer=0,
                  disk_scribble=False,
                  sample_back = False,
                  previous_scribbles = None):

    nlabels_budget = n_labels_total + 0

    # first sample order from image with less to more number of labels in total
    nlabels_total_array = np.zeros([len(train_masks)])
    for i in range(len(train_masks)):
        label_image = np.array(train_masks[i])
        nlabels_total_array[i] = np.unique(label_image).shape[0]-1
    print('total labels per sample (image): ' ,nlabels_total_array)
    ## order based on number of labels so that the last has the more number of labels if more scribbles are needed for the budget
    ix_order = np.argsort(nlabels_total_array)

    Y_out_dic = {}
    nlabels_dic = {}
    nlabels_total_dic = {}
    print('sample_i,nbudget,nscribbles')
    for ix in range(len(train_masks)):
        i = ix_order[ix]
        label_image = np.array(train_masks[i])
        n_labels_i = np.maximum(2, int(nlabels_budget / (len(train_masks) - ix)))  # budget for i

        if previous_scribbles is not None:
            previous_scribbles_i = np.array(previous_scribbles[i])
        else:
            previous_scribbles_i = None

        print(i,nlabels_budget,n_labels_i)
        mask_scribbles, nscribbles, nlabels = get_scribbles_mask(label_image, fov_box=fov_box,
                                                                 max_labels=n_labels_i,
                                                                 radius_pointer=radius_pointer,
                                                                 disk_scribble=disk_scribble,
                                                                 sample_back = sample_back,
                                                                 previous_scribbles = previous_scribbles_i)

        nlabels_budget -= nscribbles
        Y_out_dic[i]= mask_scribbles
        nlabels_dic[i] = nscribbles
        nlabels_total_dic[i] = nlabels

    Y_out = []
    nlabels_list = []
    nlabels_total_list = []

    # reorder to original ordering
    for i in range(len(train_masks)):
        Y_out.append(Y_out_dic[i])
        nlabels_list.append(nlabels_dic[i])
        nlabels_total_list.append(nlabels_total_dic[i])

    return Y_out, nlabels_list, nlabels_total_list

# def get_dataset(pd_scribbles,n_patches_per_image_train=30,n_patches_per_image_val=8,patch_size=(128, 128),
#                 p_label = 0.6,val_perc = 0.3,verbose = True, border = False):
#
#     X_train = None
#     X_val = None
#     for i in range(len(pd_scribbles)):
#
#         ## read image and label
#         npz_read = np.load(pd_scribbles['input_dir'][i] + pd_scribbles['input_file'][i])
#         image = npz_read['image']
#         label = npz_read['label']
#         nuclei = np.zeros_like(label)
#         nuclei[label > 0] = 1
#
#         ## read scribbles
#         npz_read = np.load(pd_scribbles['input_dir'][i] + pd_scribbles['scribble_file'][i])
#         scribble = npz_read['scribble']
#
#         raw_image_in = image + 0  # normalize(image,pmin=pmin,pmax=pmax,clip = False)
#
#         ## Sample validation mask
#         patch_val_size = [int(image.shape[0] * val_perc),
#                           int(image.shape[1] * val_perc)]
#         all_back = True
#         while all_back:
#
#             val_mask = np.zeros([raw_image_in.shape[0], raw_image_in.shape[1]])
#             ix_x = np.random.randint(0, raw_image_in.shape[0] - patch_val_size[0])
#             ix_y = np.random.randint(0, raw_image_in.shape[0] - patch_val_size[1])
#
#             val_mask[ix_x:ix_x + patch_val_size[0], ix_y:ix_y + patch_val_size[1]] = 1
#
#             if np.sum(val_mask * np.sum(scribble[...], axis=-1)) > 10:
#                 all_back = False
#
#         ## Generate patches
#         raw_data = RawData.from_arrays(raw_image_in[np.newaxis, ...], scribble[np.newaxis, ...])
#
#         ## for plot ##
#         if verbose:
#             aux = np.zeros([raw_image_in.shape[0], raw_image_in.shape[1], 3])
#             if len(raw_image_in.shape)>2:
#                 aux[..., 1] = np.sum(raw_image_in,axis=-1) * 0.8
#             else:
#                 aux[..., 1] = raw_image_in * 0.8
#             aux[..., 0] = scribble[..., 0]
#             aux[..., 2] = np.sum(scribble[..., 1:], axis=2)
#         ###
#
#         for group in ['val', 'train']:
#             if group == 'val':
#                 fov_mask = np.array(val_mask)
#                 n_patches_per_image = n_patches_per_image_val + 0
#                 if verbose:
#                     plt.figure(figsize=(10, 5))
#                     plt.subplot(1, 2, 1)
#                     plt.title('Validation FOV')
#                     plt.imshow(fov_mask[..., np.newaxis] * aux)
#
#             else:
#                 fov_mask = 1 - np.array(val_mask)
#                 n_patches_per_image = n_patches_per_image_train + 0
#                 if verbose:
#                     plt.subplot(1, 2, 2)
#                     plt.title('Train FOV')
#                     plt.imshow(fov_mask[..., np.newaxis] * aux)
#                     plt.show()
#
#             X_aux, Y_aux, axes = generate_patches_syxc(raw_data, patch_size,
#                                                        int(n_patches_per_image * (1 - p_label)),
#                                                        normalization=None, patch_filter=None,
#                                                        fov_mask=fov_mask)
#
#             n_patches_add = int(n_patches_per_image - X_aux.shape[0])
#
#             if n_patches_add > 0:
#                 X_labeled_aux, Y_labeled_aux, axes = generate_patches_syxc(raw_data, patch_size,
#                                                                            n_patches_add,
#                                                                            normalization=None,
#                                                                            mask_filter_index=np.arange(
#                                                                                scribble.shape[-1]),
#                                                                            fov_mask=fov_mask)
#                 if X_labeled_aux is not None:
#                     X_aux = np.concatenate([X_aux, X_labeled_aux], axis=0)
#                     Y_aux = np.concatenate([Y_aux, Y_labeled_aux], axis=0)
#
#             if group == 'val':
#                 if X_val is None:
#                     X_val = np.array(X_aux)
#                     Y_val = np.array(Y_aux)
#                 else:
#                     X_val = np.concatenate([X_val, X_aux], axis=0)
#                     Y_val = np.concatenate([Y_val, Y_aux], axis=0)
#
#             else:
#                 if X_train is None:
#                     X_train = np.array(X_aux)
#                     Y_train = np.array(Y_aux)
#                 else:
#                     X_train = np.concatenate([X_train, X_aux], axis=0)
#                     Y_train = np.concatenate([Y_train, Y_aux], axis=0)
#
#     print(Y_train.shape,Y_val.shape)
#     if border:
#         return X_train,Y_train,X_val,Y_val
#     else:
#         out_channels = int(Y_train.shape[-1]/3)
#         Y_train_aux = np.zeros([Y_train.shape[0], Y_train.shape[1], Y_train.shape[2], out_channels * 2])
#         Y_val_aux = np.zeros([Y_val.shape[0], Y_val.shape[1], Y_val.shape[2], out_channels * 2])
#         # print(out_channels,Y_train.shape[2])
#         for j in np.arange(out_channels):
#             # print(j*2,j*out_channels)
#             Y_train_aux[..., 2*j] = np.array(Y_train[..., out_channels*j])  # foreground
#             Y_train_aux[..., 2*j+1] = Y_train[..., out_channels*j+1] + Y_train[..., out_channels*j+2]  # Border + background are background
#
#             Y_val_aux[..., 2*j] = np.array(Y_val[..., out_channels*j])  # foreground
#             Y_val_aux[..., 2*j+1] = Y_val[..., out_channels*j+1] + Y_val[..., out_channels*j+2]  # Border + background are background
#         return X_train,Y_train_aux,X_val,Y_val_aux