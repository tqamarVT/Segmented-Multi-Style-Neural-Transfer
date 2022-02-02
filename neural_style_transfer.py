##################################################################################################################################################################################################################
########################################### NEURAL STYLE TRANSFER MODEL ##########################################################################################################################################
class NeuralStyleTranferModel:
    def __init__(self, styleImg1, styleImg2, contentImg, styleLayers, contentLayers, transfer_type, pool_type, mask):
        self.styleImg1 = styleImg1
        self.styleImg2 = styleImg2
        self.contentImg = contentImg        
        self.mask = mask
        self.mask_non_gray = mask
        self.setImageMask()
        self.styleLayers = styleLayers
        self.contentLayers = contentLayers
        self.numContentLayers = len(contentLayers)
        self.numStyleLayers = len(styleLayers)
        self.lossList = []
        self.results_path = sys_path + "Results/"
        # Set single style transfer or multi style transfer mode
        if(transfer_type == 'single'):
            self.single = True
        else:
            self.single = False
        # Set average or max pooling
        if(pool_type == 'max'):
          self.pool = 'max'
        else:
          self.pool = 'avg'
    ######################################################################################################################################################
    ###### METHOD TO BUILD VGG MODEL FOR FEATURE SPACE EXTRACTION ########################################################################################            
    def buildModel(self):
        if(self.pool == 'max'):
          # initialize model and exclude FC layers
          fe = VGG19(include_top=False, weights='imagenet')          
          # define model inputs and outputs        
          tempModel = Model(inputs=fe.input, outputs=[fe.get_layer(layer).output for layer in self.contentLayers+self.styleLayers])          
          # set all network layers to not be trained
          tempModel.trainable = False            
          self.model =  tempModel
          #self.model.summary()
        else:
          vgg = VGG19(include_top=False, weights='imagenet')
          fe = self.replace_max_by_average_pooling(vgg)
          tempModel = Model(inputs=fe.input, outputs=[fe.get_layer(layer).get_output_at(1) for layer in self.contentLayers+self.styleLayers])
          # set all network layers to not be trained
          tempModel.trainable = False            
          self.model =  tempModel
          #self.model.summary()
        h = self.contentImg.shape[1]
        w = self.contentImg.shape[2]
        data_augmentation = tf.keras.Sequential([
        Layers.experimental.preprocessing.RandomCrop(height = h, width = w),
        Layers.experimental.preprocessing.RandomRotation(0.01),])
        self.aug_model = data_augmentation
        
    def replace_max_by_average_pooling(self, model):
        model = VGG19(include_top=False, weights='imagenet')
        input_layer, *other_layers = model.layers
        assert isinstance(input_layer, keras.layers.InputLayer)
        
        x = input_layer.output
        for layer in other_layers:
            if isinstance(layer, keras.layers.MaxPooling2D):
                layer = keras.layers.AveragePooling2D(pool_size=(layer.pool_size), strides=layer.strides, padding=layer.padding, data_format=layer.data_format,
                name=f"{layer.name}_av",)
            x = layer(x)
        newModel = Model(inputs=input_layer.input, outputs=x)
        return newModel
    ######################################################################################################################################################
    ###### METHOD TO EXTRACT STYLE/CONTENT FEATURES OF IMAGES AND CALCULATE GRAM MATRICES ################################################################
    def extractFeatures(self, output_only):
    
        #Extract FeatureMaps for the Output Image Only (Updated during Style Transfer)
        if(output_only == True):
            # Repeat the process for the output image
            tempOutputImage = self.outputImage
            ##
            #self.setImageMask()
            ##
            tempOutputImage = preprocessInput(self.outputImage)
            tempOutputImage = self.aug_model(tempOutputImage)
            outputImageFeatures = self.model(tempOutputImage)
            self.outputStyleFeatures = {styleName: value for styleName, value in zip(self.styleLayers, outputImageFeatures[self.numContentLayers:])}
            ##
            ###for styleLayer in self.outputStyleFeatures:
                ###length = self.outputStyleFeatures[styleLayer].shape[1]
                ###width = self.outputStyleFeatures[styleLayer].shape[2]
                ###volume = self.outputStyleFeatures[styleLayer].shape[3]
                ###temp_mask = cv2.resize(self.mask, (width,length), interpolation = cv2.INTER_AREA)
                ###mask_volume = np.repeat(temp_mask[...,None],volume,axis=2)
                ###self.outputStyleFeatures[styleLayer] = tf.math.multiply(self.outputStyleFeatures[styleLayer], mask_volume)                
            ##
            self.outputContentFeatures = {contentName: value for contentName, value in zip(self.contentLayers, outputImageFeatures[:self.numContentLayers])}     
            return
            
            
        # Extract features and calculate gram matrix for single-style transfer
        if(self.single):
            # Starting with the style image
            styleImg1 = preprocessInput(self.styleImg1)
            # Get feature maps from content and style layers
            self.buildModel()
            styleImgFeatures1 = self.model(styleImg1)
            # create dictionary of style layer name and its corresponding feature maps for easier access
            styleFeatures1 = {styleName: value for styleName, value in zip(self.styleLayers, styleImgFeatures1[self.numContentLayers:])}
            #### TEMPORARY STYLE FEATURE MAP MASKING
            #for styleLayer in styleFeatures1:
                #length = styleFeatures1[styleLayer].shape[1]
                #width = styleFeatures1[styleLayer].shape[2]
                #volume = styleFeatures1[styleLayer].shape[3]
                #temp_mask = cv2.resize(self.mask, (width,length), interpolation = cv2.INTER_AREA)
                #mask_volume = np.repeat(temp_mask[...,None],volume,axis=2)
                #styleFeatures1[styleLayer] = tf.math.multiply(styleFeatures1[styleLayer], mask_volume)                
            #### TEMPORARY STYLE FEATURE MAP MASKING
            self.styleFeatures1 = styleFeatures1
            self.styleGramMatrices1 = {styleLayer: self.getGramMatrix(featureMaps) for styleLayer, featureMaps in self.styleFeatures1.items()}
            self.styles = {'styleImg1': (self.styleGramMatrices1, 1.0)}

            # Repeat the process for the content image
            contentImg = preprocessInput(self.contentImg)
            contentImgFeatures = self.model(contentImg)
            # create dictionary of content layer name and its corresponding feature maps for easier access
            contentFeatures = {contentName: value for contentName, value in zip(self.contentLayers, contentImgFeatures[:self.numContentLayers])}
            self.contentFeatures = contentFeatures
            
            # Repeat the process for the output image
            self.outputImage = tf.Variable(self.contentImg)
            tempOutputImage = preprocessInput(self.outputImage)
            outputImageFeatures = self.model(tempOutputImage)
            self.outputStyleFeatures = {styleName: value for styleName, value in zip(self.styleLayers, outputImageFeatures[self.numContentLayers:])}
            ##
            ###for styleLayer in self.outputStyleFeatures:
                ###length = self.outputStyleFeatures[styleLayer].shape[1]
                ###width = self.outputStyleFeatures[styleLayer].shape[2]
                ###volume = self.outputStyleFeatures[styleLayer].shape[3]
                ###temp_mask = cv2.resize(self.mask, (width,length), interpolation = cv2.INTER_AREA)
                ###mask_volume = np.repeat(temp_mask[...,None],volume,axis=2)
                ###self.outputStyleFeatures[styleLayer] = tf.math.multiply(self.outputStyleFeatures[styleLayer], mask_volume)                
            ##
            self.outputContentFeatures = {contentName: value for contentName, value in zip(self.contentLayers, outputImageFeatures[:self.numContentLayers])}            
            
        # Extract features and calculate gram matrices for multi-style transfer    
        if(not self.single):
            # Starting with the style image one
            styleImg1 = preprocessInput(self.styleImg1)
            # Get feature maps from content and style layers
            self.buildModel()
            styleImgFeatures1 = self.model(styleImg1)
            # create dictionary of style layer name and its corresponding feature maps for easier access
            styleFeatures1 = {styleName: value for styleName, value in zip(self.styleLayers, styleImgFeatures1[self.numContentLayers:])}
            self.styleFeatures1 = styleFeatures1
            
            # Starting with the style image two
            styleImg2 = preprocessInput(self.styleImg2)
            # Get feature maps from content and style layers            
            styleImgFeatures2 = self.model(styleImg2)
            # create dictionary of style layer name and its corresponding feature maps for easier access
            styleFeatures2 = {styleName: value for styleName, value in zip(self.styleLayers, styleImgFeatures2[self.numContentLayers:])}
            self.styleFeatures2 = styleFeatures2
            
            self.styleGramMatrices1 = {styleLayer: self.getGramMatrix(featureMaps) for styleLayer, featureMaps in self.styleFeatures1.items()}
            self.styleGramMatrices2 = {styleLayer: self.getGramMatrix(featureMaps) for styleLayer, featureMaps in self.styleFeatures2.items()}
            self.styles = {'styleImg1': (self.styleGramMatrices1, 0.5), 'styleImg2': (self.styleGramMatrices2, 0.5)}
       
            # Repeat the process for the content image
            contentImg = preprocessInput(self.contentImg)
            contentImgFeatures = self.model(contentImg)
            # create dictionary of content layer name and its corresponding feature maps for easier access
            contentFeatures = {contentName: value for contentName, value in zip(self.contentLayers, contentImgFeatures[:self.numContentLayers])}
            self.contentFeatures = contentFeatures            
            
            # Repeat the process for the output image
            self.outputImage = tf.Variable(self.contentImg)
            tempOutputImage = preprocessInput(self.outputImage)
            outputImageFeatures = self.model(tempOutputImage)
            self.outputStyleFeatures = {styleName: value for styleName, value in zip(self.styleLayers, outputImageFeatures[self.numContentLayers:])}
            self.outputContentFeatures = {contentName: value for contentName, value in zip(self.contentLayers, outputImageFeatures[:self.numContentLayers])}            
    
    ######################################################################################################################################################
    ###### METHOD TO CALCULATE GRAM MATRIX ON FEATUREMAPS ############################################################################################
    def getGramMatrix(self, featureMaps):
        # calcualte feature correlations
        gramMatrix = tf.einsum('abcd,abce->ade', featureMaps, featureMaps)
        input_shape = tf.shape(featureMaps)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return gramMatrix/num_locations
    ######################################################################################################################################################
    ###### METHOD TO OBTAIN AND DISPLAY FEATURE MAPS #####################################################################################################
    # def displayFeatureMaps(self):
    #     imageOne = 'Style Image One'
    #     imageTwo = 'Style Image Two'
    #     if(self.single):
    #       for layer, features in self.styleFeatures1.items():
    #         plt.imshow(features[0, :, :, 0], cmap='gray')
    #         plt.axis('off')
    #         plt.title(imageOne + ': ' + layer )
    #         plt.show()            
    #     else:
    #       for layer, features in self.styleFeatures1.items():
    #         plt.imshow(features[0, :, :, 0], cmap='gray')
    #         plt.axis('off')
    #         plt.title(imageOne + ': ' + layer )
    #         plt.show()            
    #       for layer, features in self.styleFeatures2.items():
    #         plt.imshow(features[0, :, :, 0], cmap='gray')
    #         plt.axis('off')
    #         plt.title(imageTwo + ': ' + layer )
    #         plt.show()                
    ######################################################################################################################################################
    ######### METHODS TO CALCULATE THE TOTAL CONTENT/STYLE LOSS FOR THE STYLE TRANSFER MODEL #############################################################
    def styleLayerLoss(self, layerFeatureMaps, styleGramMatrix):
        #using formula from original paper Gatys et al.
        #height, width, numChannels = layerFeatureMaps.get_shape().as_list()        
        gram_style = self.getGramMatrix(layerFeatureMaps)        
        return tf.reduce_mean(tf.square(gram_style - styleGramMatrix))
        #return tf.reduce_mean(tf.abs(gram_style - styleGramMatrix))
            
    def contentLoss(self, layerFeatureMaps, contentFeatureMaps):  
      return tf.reduce_mean(tf.square(layerFeatureMaps - contentFeatureMaps))
      #return tf.reduce_mean(tf.abs(layerFeatureMaps - contentFeatureMaps))
                  
    def totalLoss(self):
        individualStyleImageLosses = []
        for style, (styleGramMatrices, imgWeight) in self.styles.items():
            individualStyleLayerLosses = []            
            for (styleLayer, layerWeight) in self.weightedStyleLayers[style].items():                                
                individualStyleLayerLosses.append(self.styleLayerLoss(self.outputStyleFeatures[styleLayer], styleGramMatrices[styleLayer])*layerWeight)
            individualStyleImageLosses.append(tf.add_n(individualStyleLayerLosses)*imgWeight)
        styleLoss = sum(individualStyleImageLosses)                      
        styleLoss *= self.styleWeight/self.numStyleLayers        
        contLoss = tf.add_n([self.contentLoss(self.outputContentFeatures[contentLayer], self.contentFeatures[contentLayer]) for contentLayer in self.contentLayers])        
        contLoss *= self.contentWeight
        self.loss = styleLoss + contLoss
    ######################################################################################################################################################
    ######################################################################################################################################################
    def styleTransfer(self, epochs, steps_per_epoch, total_variation_weight):
        for epoch in range(epochs):
            print("===========================================================================================================================================================")
            print("\nStart of epoch %d" % (epoch,))
            print("===========================================================================================================================================================")
            step = 0    
            #self.alphaBlending()
            for m in range (steps_per_epoch):
                step += 1
                with tf.GradientTape() as tape:
                    tape.watch(self.outputImage)    
                    self.extractFeatures(output_only=True)
                    self.totalLoss()
                    #print(self.loss)
                    # add total variation to denoise and get smoother result
                    self.loss += total_variation_weight*tf.image.total_variation(self.outputImage)
                    self.lossList.append(self.loss)
                grads = tape.gradient(self.loss, self.outputImage)
                self.optimizer.apply_gradients([(grads, self.outputImage)])
                self.outputImage.assign(tf.clip_by_value(self.outputImage, 0, 1))
                #print("step: ", step)        
                #display.clear_output(wait=True)
                #display.display(tensor_to_image(imgVar))
                #print("Train step: {}".format(step))
        #self.alphaBlending()
        self.img = tensor_to_image(self.outputImage)    
        display.display(self.img)                    
    ######################################################################################################################################################
    ################# METHOD TO INITLIAZE MODEL WITH DEFAULT PARAMETERS ##################################################################################
    def initialize_default(self):
      imgOneSLW = [9, 9, 9, 3, 3]
      imgTwoSLW = [9, 9, 9, 3, 3]  
      self.extractFeatures(output_only=False)
      self.setGlobalWeights(style_weight = 1e-2, content_weight = 1e4)
      self.setStyleLayerWeights(imgOneSLW, imgTwoSLW)
      self.setImageWeights(0.5, 0.5)
      self.setOptimizer(non_default=False, learning_rate = 0, beta_1 = 0, epsilon = 0)
      #self.displayFeatureMaps()      
    ######################################################################################################################################################
    ################# METHOD TO INITLIAZE MODEL FOR NONDEFAULT PARAMETERS ################################################################################
    def initialize_nondefault(self):
      self.extractFeatures(output_only=False)    
      print("Non-default initialization will require you to set global weights, set style layer weights, set image 1 to n weights (for multi-style transfer), and set optimizer parameters \n")
      #self.displayFeatureMaps()      
    ######################################################################################################################################################
    ######################################################################################################################################################
    def run(self, epochs, steps_per_epoch, total_variation_weight):
        #self.runNumber = run_number
        self.variation = total_variation_weight
        self.setOptimizer(non_default=False, learning_rate = 0, beta_1 = 0, epsilon = 0)
        start = time.time()
        self.styleTransfer(epochs, steps_per_epoch, total_variation_weight)
        end = time.time()
        print("Total time: {:.1f}".format(end-start))
        self.saveResults()
        return self.img
    ######################################################################################################################################################
    ###### METHOD TO SET OPTIMIZER AND PARAMETERS ########################################################################################################
    def setOptimizer(self, non_default, learning_rate, beta_1, epsilon):
      if(non_default):
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, epsilon=epsilon)    
      else: 
        self.optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)    
  ######################################################################################################################################################
  ###### METHOD TO ADJUST STYLE/CONTENT WEIGHTS  #######################################################################################################
    def setGlobalWeights(self, style_weight, content_weight):
      self.styleWeight = style_weight
      self.contentWeight = content_weight
  #####################################################################################################################################################
  #####################################################################################################################################################
    def setStyleLayerWeights(self, styleImgOneLayerWeights, styleImgTwoLayerWeights):
        imageOneWeightedStyleLayers = {styleName: value for styleName, value in zip(self.styleLayers, styleImgOneLayerWeights)}
        imageTwoWeightedStyleLayers = {styleName: value for styleName, value in zip(self.styleLayers, styleImgTwoLayerWeights)}
        #self.weightedStyleLayers = {'styleImg1': (imageOneWeightedStyleLayers), 'styleImg2': (imageTwoWeightedStyleLayers)}        
        if(self.single):
          self.weightedStyleLayers = {'styleImg1': (imageOneWeightedStyleLayers)}        
        else:
          self.weightedStyleLayers = {'styleImg1': (imageOneWeightedStyleLayers), 'styleImg2': (imageTwoWeightedStyleLayers)}        
  #####################################################################################################################################################                 
  ######################################################################################################################################################
    def setImageWeights(self, img1_weight, img2_weight):
      if(self.single):
        self.styles = {'styleImg1': (self.styleGramMatrices1, img1_weight)}        
      else:
        self.styles = {'styleImg1': (self.styleGramMatrices1, img1_weight), 'styleImg2': (self.styleGramMatrices2, img2_weight)}        
  ###### METHOD TO GET CURRENT MODEL SUMMARY  ##########################################################################################################
    def getModelSummary(self):
      self.model.summary()
  ###### METHOD TO SAVE RESULTS WITH DETAILS  ##########################################################################################################
    def saveResults(self):
      image = self.img
      results_path = self.results_path
      image.save(os.path.join(results_path + "contentimage2" + ".png"))
      np.set_printoptions(suppress=True)
      loss = np.array(self.lossList, dtype=np.float32)
      # plt.plot(loss)
      # plt.title('model loss')
      # plt.ylabel('loss')
      # plt.xlabel('epoch')
      # plt.savefig(os.path.join(results_path + "lossPlot" + ".jpeg"))
      # plt.close()
      with open(results_path + "parameters_" + ".txt", 'w') as writefile:
        writefile.write('LAYER WEIGHTS \n')
        for (styleLayer, layerWeight) in self.weightedStyleLayers.items():
          writefile.write(' ' + styleLayer + ": " + str(layerWeight))
        writefile.write('\n')
        writefile.write('IMAGE WEIGHTS  \n')
        for style, (styleGramMatrices, imgWeight) in self.styles.items():
          writefile.write(' ' + style + ": " + str(imgWeight))
        writefile.write('\n')             
        writefile.write('GLOBAL WEIGHTS \n')
        writefile.write('style weight: ' + str(self.styleWeight) + '\n')
        writefile.write('content weight: ' + str(self.contentWeight) + '\n')  
        writefile.write('ALPHA/BETA RATIO: ' + str(self.styleWeight/self.contentWeight) + '\n')
        writefile.write('Total variation weight: ' + str(self.variation) + '\n')
        writefile.write('Pooling Type: ' + self.pool + '\n')
  ####### METHOD TO SET NST MODEL MASK FOR LOCALIZED STYLE TRANSFER #######################################################################################                 
    def setImageMask(self):    
      self.mask = cv2.cvtColor(self.mask, cv2.COLOR_RGB2GRAY) # Convert the mask to grayscale
      self.mask = self.mask.astype(int)/255 
      #self.mask = (self.mask//255).astype('float64')  
  ####### METHOD TO BLEND CONTENT IMAGE WITH SELF.OUTPUTIMAGE #######################################################################################                 
    def alphaBlending(self):
        source = tensor_to_image(self.outputImage)
        source = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)
        target = tensor_to_image(self.contentImg)
        target = cv2.cvtColor(np.array(target), cv2.COLOR_RGB2BGR)
        mask = cv2.cvtColor(self.mask_non_gray, cv2.COLOR_RGB2BGR)
        mask = cv2.resize(mask, (source.shape[1],source.shape[0]), interpolation = cv2.INTER_AREA)
        foreground = source
        background = target
        alpha = mask
        foreground = foreground.astype(float)
        background = background.astype(float)
        alpha = alpha.astype(float)/255
        foreground = cv2.multiply(alpha, foreground)
        background = cv2.multiply(1.0 - alpha, background)
        outImage = cv2.add(foreground, background)
        cv2.imwrite(self.results_path + 'temp_self_output_image.png', outImage);
        #outImage = outImage.astype(np.float32)
        #outImage = cv2.cvtColor(outImage, cv2.COLOR_BGR2RGB)
        #outImage = Image.fromarray(outImage)
        #tempOutputImage = self.loadBlendedImage(outImage)
        tempOutputImage = self.loadBlendedImage(self.results_path + 'temp_self_output_image.png')
        self.outputImage = tf.Variable(tempOutputImage)
        
    def loadBlendedImage(self, path_to_img):
      max_dim = 666
      img = tf.io.read_file(path_to_img)
      img = tf.image.decode_image(img, channels=3)
      img = tf.image.convert_image_dtype(img, tf.float64)
      shape = tf.cast(tf.shape(img)[:-1], tf.float64)
      long_dim = max(shape)
      scale = max_dim / long_dim
      new_shape = tf.cast(shape * scale, tf.int32)
      img = tf.image.resize(img, new_shape)
      img = img[tf.newaxis, :]
      return img
