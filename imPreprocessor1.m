function im_out = imPreprocessor1(filename)

x = 'vgg16'; % alexnet, vgg16,inceptionv3
im_out = imread(filename);

switch x

    case 'alexnet'
        im_out = imresize(im_out,[227,227]);
        
    case 'vgg16'
        im_out = imresize(im_out,[224,224]);
        
    case 'inceptionv3'
        im_out = imresize(im_out,[299,299]);

end

end
