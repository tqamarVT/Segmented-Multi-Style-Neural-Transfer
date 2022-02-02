# Multi style transfer
contentLayers = ['block5_conv2']
styleLayers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

styleImagePath1 = 'Data/Style Images/wood_style.jpg'
styleImagePath2 = 'Data/Style Images/The_Great_Wave_off_Kanagawa.jpg'
contentImagePath1 = 'Data/Content Images/contentimage1.png'


styleImage1 = load_img(sys_path + styleImagePath1)
styleImage2 = load_img(sys_path + styleImagePath2)
contentImage1 = load_img(sys_path + contentImagePath1)



run = 0
pool_type = ['avg']
variation = [0,10,20,30,40,50,60]
gbw= []
gbw.append([1e-2, 1e4])
gbw.append([1e-1, 1e4])
imgOneSLW = [9, 9, 9, 3, 3]
imgTwoSLW = [9, 9, 9, 3, 3]  
imw = [0.5, 0.5]

base_url = 'http://images.cocodataset.org/val2017/'
img_type = '.jpg'
url = base_url + '000000054123' + img_type
#url = 'https://www.thebigjourneycompany.com/uploads/news-images/3213048-highland-cow.jpg'

#################################################### FIRST SEGMENTED STYLE TRANSFER ###################################################################################################################################
#panop_seg_model = PanopticSegmentationModel(image_path = url, image_name = "000000007281", image_path_local = False)
#segmentation_range = chain(range(8,9))
#panop_seg_model.set_panoptic_segmentation_range(segmentation_range)
#segmented_mask = panop_seg_model.run()
#segmentedContentImagePath = 'Data/Segmented Content Images/000000007281.jpg'
#segmentedContentImage = load_img(sys_path + segmentedContentImagePath)


#for i in range(len(pool_type)):
  #with tf.device('/gpu:0'):
    #nstModel = NeuralStyleTranferModel(styleImg1 = styleImage10, styleImg2 = styleImage2, contentImg = segmentedContentImage, styleLayers = styleLayers, contentLayers = contentLayers, 
                                       #transfer_type = "single", pool_type = pool_type[i])
    #nstModel.initialize_default()
    #nstModel.setGlobalWeights(style_weight = 1e1, content_weight = 1e3)
    #stylized_image = nstModel.run(epochs=20, steps_per_epoch=10, total_variation_weight=variation[1])
    #stylized_image = nstModel.run(epochs=20, steps_per_epoch=10, total_variation_weight=variation[1])
    #stylized_image.show()
    
    #final_image_one = panop_seg_model.blend_segmented_mask(stylized_image = stylized_image)
    #final_image_one.save(os.path.join(sys_path + "Results/" + "stylizedImage_winterize" + ".png"))
    #stylized_image_one = Image.open(sys_path + "Results/" + "stylizedImage_winterize" + ".png")

#################################################### SECPMD SEGMENTED STYLE TRANSFER ###################################################################################################################################
#panop_seg_model = PanopticSegmentationModel(image_path = url, image_name = "000000031093_1", image_path_local = False)
#segmentation_range = chain(range(1,2))
#panop_seg_model.set_panoptic_segmentation_range(segmentation_range)
#segmented_mask = panop_seg_model.run()
#segmentedContentImagePath2 = 'Data/Segmented Content Images/000000031093_1.png'
#segmentedContentImage2 = load_img(sys_path + segmentedContentImagePath2)

#for i in range(len(pool_type)):
  #with tf.device('/gpu:0'):
    #nstModel = NeuralStyleTranferModel(styleImg1 = styleImage8, styleImg2 = styleImage2, contentImg = segmentedContentImage2, styleLayers = styleLayers, contentLayers = contentLayers, 
                                       #transfer_type = "single", pool_type = pool_type[i])
    #nstModel.initialize_default()
    #nstModel.setGlobalWeights(style_weight = 1e-1, content_weight = 1e3)
    #stylized_image_two = nstModel.run(epochs=20, steps_per_epoch=10, total_variation_weight=variation[1])
    #stylized_image.show()

    #panop_seg_model.change_base_image(final_image_one)
    #final_image = panop_seg_model.blend_segmented_mask(stylized_image = stylized_image_two)
    #final_image_one.save(os.path.join(sys_path + "Results/" + "finalStylizedImage" + ".png"))




panop_seg_model = PanopticSegmentationModel(image_path = url, image_name = "000000054123", image_path_local = False)
segmentation_range = chain(range(2,7))
panop_seg_model.set_panoptic_segmentation_range(segmentation_range)
segmented_picture = panop_seg_model.run()
segmentedContentImagePath = 'Data/Segmented Content Images/000000054123.jpg'
#segmentedContentImage = load_img(sys_path + 'segmentedContentImagePath')
mask = panop_seg_model.return_current_mask()



for i in range(len(pool_type)):
  with tf.device('/gpu:0'):
    nstModel = NeuralStyleTranferModel(styleImg1 = styleImage1, styleImg2 = styleImage2, contentImg = contentImage1, styleLayers = styleLayers, contentLayers = contentLayers, 
                                       transfer_type = "single", pool_type = pool_type[i], mask = mask)
    nstModel.initialize_default()
    nstModel.setGlobalWeights(style_weight = 1e1, content_weight = 1e3)
    nstModel.setStyleLayerWeights(imgOneSLW, imgTwoSLW)
    contentimage0 = nstModel.run(epochs=40, steps_per_epoch=10, total_variation_weight=variation[1])
    
  
