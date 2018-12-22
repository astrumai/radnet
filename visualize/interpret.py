""""
I need to write a class for filter visualizer. Robust to different depths and capable of outputting different types
of maps based on users choice.

The user will call a function outside of the class, like the aguments and specify the output image type and the depth
or specific depth he wants to visualize

Notes:
    Depending on teh depth I will have to resize the image strarting from lets' say we start at 64 > 128 > 256
    The downsample and upsample block has different types of blocks so there has to be two types of functions for those
    as well.
"""


# split the down sampling blocks
def split_up_blocks(layer, depth):
    block_list = []
    for d in depth:
        temp = list(layer.children())[d]
        block_list.append(temp)
    return block_list


# split the down sampling blocks
def split_down_blocks(layer, depth):
    block_list = []
    for d in depth:
        temp = list(layer.children())[d]
        block_list.append(temp)
    return block_list


# a function to pick the blocks for interpreting
def blocks(sampling, depth, model, args):

    layers = list(model.children())
    # chose between downsampling or upsampling

    if sampling == 'down':
        # if downsampling: call a function that takes in the depth of the model and splits it into respective blocks
        layer = layers[0]
        assert depth <= args.depth, "Depth should be equal to or less than the u-net depth"
        block_list = split_down_blocks(layer, depth)
        return block_list

    elif sampling == 'up':
        # if upsampling: do the same but will have to add another child into it since the blocks has upsampling in it
        # need to see how to figure that out
        layer = layers[1]
        assert depth <= (args.depth-1), "Up sampling depth should be one less than the u-net depth"
        block_list = split_up_blocks(layer, depth)
        return block_list

    elif sampling == 'both':
        # if both call both sequentially

        return


# a function to load the image
def load_test_image(image, ):
    # the user specifies the path to the image

    # read image path

    # load the image based on the image path

    # do resize and to tensor depending on the conv block

    # squeeze the output and send it to the cuda device

    pass


# load different blocks for predicting
def load_blocks():
    # load different block for predicting

    pass


# plot the weights
def plot_filters():
    # choose between different blocks and pick which one to plot from

    pass


# main function the caller will use
def interpret(args):
    # load trained model, if model is not available raise error and call for training

    # layers = list(model.children())
    # l = layers[0]
    # tem = list(l.children())[1]
    # tem2 = list(tem.children())
    # print(tem)
    #
    # # from visualize.interpret import layer_outputs
    #
    # kit = Image.fromarray((tiff.imread(trainPath))[1])
    # _data = Compose([Resize(64), ToTensor()])
    # kit2 = _data(kit)
    # kit3 = kit2.unsqueeze(0).to(device)
    #
    # image_vis = tem(kit3)
    # print(image_vis.shape)
    # print(image_vis[0][0].shape)

    #
    # fig = plt.figure()
    # plt.rcParams["figure.figsize"] = (128, 128)
    # """The output of the 1st block has 64 filters so by changing from [0][0 to 63] you can visualize those filters"""
    #
    # for i in range(64):
    #     fig.add_subplot(8, 8, i+1)
    #     imgplot = plt.imshow(image_vis[0][i].cpu().detach().numpy())
    #     plt.axis('off')
    # plt.show()
    pass
