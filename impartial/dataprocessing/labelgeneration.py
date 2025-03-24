import numpy as np
from skimage import morphology
from scipy import ndimage


def get_scribbles_mask(label_image, fov_box=(32, 32), max_labels=4,
                       radius_pointer=0, disk_scribble=False,sample_back = False):
    mask_image = np.zeros_like(label_image)
    mask_image[label_image > 0] = 1.0

    nlabels = np.unique(label_image[label_image > 0]).shape[0]
    max_labels = np.minimum(max_labels, nlabels)

    ### Set the instance scribble mask mask_sk
    mask_sk = morphology.skeletonize(mask_image) * mask_image
    if radius_pointer > 0:
        selem = morphology.disk(radius_pointer)
        mask_sk = morphology.dilation(mask_sk, footprint=selem) * mask_image
    if disk_scribble:
        selem = morphology.disk(3)
        outline_arc = morphology.erosion(mask_image, footprint=selem) - \
                      morphology.erosion(morphology.erosion(mask_image, footprint=selem), footprint=morphology.disk(1))
        if radius_pointer > 0:
            selem = morphology.disk(radius_pointer)
            outline_arc = morphology.dilation(outline_arc, footprint=selem)

        mask_sk += outline_arc
        mask_sk[mask_sk > 0] = 1
        mask_sk = mask_sk * mask_image

    nbudget = max_labels + 0
    labels_image_res = np.array(label_image)

    back_aux = np.array(mask_image)
    back_aux = morphology.dilation(back_aux, footprint=morphology.disk(4))

    fov_image_res = np.array(1-mask_image)*back_aux #background entre foreground
    fov_image_res = morphology.skeletonize(fov_image_res)

    # plt.figure(figsize=(10,5))
    # plt.subplot(1,2,1)
    # plt.imshow(fov_image_res)
    # plt.subplot(1, 2, 2)

    back_aux = morphology.skeletonize(1-back_aux)
    fov_image_res += morphology.dilation(back_aux,footprint=morphology.disk(4))

    # plt.imshow(fov_image_res)
    # plt.show()

    foreground = np.zeros_like(labels_image_res)
    background = np.zeros_like(labels_image_res)

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

                back_aux = morphology.dilation(label_aux, footprint=morphology.disk(4))
                back_aux += bb_image
                back_aux = back_aux * (1 - mask_image)
                back_aux[back_aux > 0] = 1
                back_aux = morphology.skeletonize(back_aux)

                if radius_pointer > 0:
                    selem = morphology.disk(radius_pointer)
                    back_aux = morphology.dilation(back_aux, footprint=selem) * (1 - mask_image)

                background += back_aux * (1 - mask_image)  # double secure

                nbudget -= 1
                labels_image_res = labels_image_res * (1 - label_aux)
                fov_image_res = fov_image_res * (1 - label_aux)
        else:
            #print('NO labels')
            if (np.sum(fov_image_res*bb_image)>0):
                back_aux = morphology.skeletonize(bb_image*(1-mask_image)*fov_image_res) ## box with background
                if radius_pointer > 0:
                    selem = morphology.disk(radius_pointer)
                    back_aux = morphology.dilation(back_aux, footprint=selem) * (1 - mask_image)
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
        mask_scribbles = np.zeros([mask_image.shape[0], mask_image.shape[1], 2])
        mask_scribbles[..., 0] = np.array(foreground)  # fore
        mask_scribbles[..., 1] = np.array(background)   # back

    return mask_scribbles, max_labels - nbudget, nlabels


def get_scribbles(train_masks, n_labels_total, fov_box=(32, 32),
                  radius_pointer=0,
                  disk_scribble=False,
                  sample_back = False):

    nlabels_budget = n_labels_total + 0

    # first sample order from image with less to more number of labels in total
    nlabels_total_array = np.zeros([len(train_masks)])
    for i in range(len(train_masks)):
        label_image = np.array(train_masks[i])
        nlabels_total_array[i] = np.unique(label_image).shape[0]-1
    print('total labels per sample (image): ' ,nlabels_total_array)
    ix_order = np.argsort(nlabels_total_array)

    Y_out_dic = {}
    nlabels_dic = {}
    nlabels_total_dic = {}
    print('sample_i,nbudget,nscribbles')
    for ix in range(len(train_masks)):
        i = ix_order[ix]
        label_image = np.array(train_masks[i])
        n_labels_i = np.maximum(2, int(nlabels_budget / (len(train_masks) - ix)))  # budget for i
        print(i,nlabels_budget,n_labels_i)
        mask_scribbles, nscribbles, nlabels = get_scribbles_mask(label_image, fov_box=fov_box,
                                                                 max_labels=n_labels_i,
                                                                 radius_pointer=radius_pointer,
                                                                 disk_scribble=disk_scribble,
                                                                 sample_back = sample_back)
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



# 2024-07-10


def erosion_labels(label, radius_pointer=1):

    selem = morphology.disk(radius_pointer)

    mask = np.zeros_like(label)

    for vlabel in np.unique(label):
        if vlabel != 0:  
            mask_label = np.zeros_like(label)
            mask_label[label == vlabel] = 1
            aux = morphology.dilation(mask_label, footprint=selem)
            if np.unique(aux*label).shape[0] > 2: #overlaps with other label
                erode_label = morphology.erosion(mask_label, footprint=selem)
                mask[erode_label>0] = 1
            else:
                mask[mask_label>0] = 1
                
    ## Image Corners
    # mask[:,0:radius_pointer] = label[:,0:radius_pointer]
    # mask[:,-radius_pointer-1:] = label[:,-radius_pointer-1:]
    # mask[0:radius_pointer,:] = label[0:radius_pointer,:]
    # mask[-radius_pointer-1:,:] = label[-radius_pointer-1:,:]

    return mask


def get_scribble(label):

    total_labels = np.unique(label)
    nlabels_budget = 30
    label_mask = erosion_labels(label, radius_pointer=1)

    Y_gt_train_ch0_list = []
    Y_gt_train_ch0_list.append(label_mask)
    
    Y_out_ch0_list, nscribbles_ch0_list, nlabels_ch0_list = get_scribbles(Y_gt_train_ch0_list,
                                                                     nlabels_budget,
                                                                     fov_box=(32,32),
                                                                     radius_pointer=0,
                                                                    #  disk_scribble = False,
                                                                    #  sample_back = False)
                                                                     disk_scribble = True,
                                                                     sample_back = True)

    
    ## concaticating scribbles :
    label_gt = Y_gt_train_ch0_list[0]
    iscribble = Y_out_ch0_list[0] 

    label_ch = np.array(label_gt)        
    background = np.array(iscribble[...,1])*(1-label_ch) + 0 #make sure no foreground is set as background
    scribble_task = np.array(iscribble[...,0])[...,np.newaxis] + 0
            
    scribble_task = np.concatenate([scribble_task,background[...,np.newaxis]],axis = -1)
    scribble_task[scribble_task>0] = 1
        
    scribble = np.array(scribble_task)
    print("scribble shape",  scribble.shape)

    return scribble


def get_fov_mask(image, scribble):

    np.random.seed(44)

    val_perc = 0.4

    region_val_size = [int(image.shape[0] * val_perc/2),int(image.shape[1] * val_perc/2)] #validation region
    mask_scribbles = np.sum(scribble,axis = -1)
    mask_scribbles[mask_scribbles>0] = 1
    mask_scribbles = ndimage.convolve(mask_scribbles, np.ones([5,5]), mode='constant', cval=0.0)

    val_center = np.random.multinomial(1, mask_scribbles.flatten()/np.sum(mask_scribbles.flatten()), size=1).flatten()
    ix_center = np.argmax(val_center)
    ix_row = int(np.floor(ix_center/image.shape[1]))
    ix_col = int(ix_center - ix_row * image.shape[1])
    print(ix_center,ix_row,ix_col)

    row_low = np.maximum(ix_row-region_val_size[0],0)
    row_high = np.minimum(row_low+region_val_size[0], image.shape[0])
    row_low = np.maximum(row_high - 2*region_val_size[0],0)
    row_high = np.minimum(row_low+ 2*region_val_size[0], image.shape[0])

    col_low = np.maximum(ix_col-region_val_size[1],0)
    col_high = np.minimum(col_low+region_val_size[1], image.shape[1])
    col_low = np.maximum(col_high - 2*region_val_size[1],0)
    col_high = np.minimum(col_low+2*region_val_size[1], image.shape[1])
    print(row_low,row_high,col_low,col_high)

    validation_mask = np.zeros([image.shape[0], image.shape[1]])
    validation_mask[row_low:row_high,
                    col_low:col_high] = 1

    return validation_mask 
