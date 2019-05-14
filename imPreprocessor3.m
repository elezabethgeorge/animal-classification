function im_out = imPreprocessor3(filename)

x = 'dleaf'; % alexnet, vgg16,dleaf
im_out = imread(filename);

switch x

    case 'alexnet'
        im_out = imresize(im_out,[227,227]);
        
    case 'vgg16'
        im_out = imresize(im_out,[224,224]);
        
    case 'dleaf'
        im_out = imresize(im_out,[169 169 ]);
        im_out = rgb2gray(im_out);

end

end
