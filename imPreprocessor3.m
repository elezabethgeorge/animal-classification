function im_out = imPreprocessor3(filename);
%filename = 'bear.jpg';
x = 'Dleaf'; % alexnet, vgg16,inceptionv3
im_out = imread(filename);
im=rgb2gray(im_out);

switch x

    case 'alexnet'
        im_out = imresize(im,[227,227]);
        montage({im,im_out});

    case 'vgg16'
        im_out = imresize(im,[224,224]);
        
   case 'Dleaf'
        im_out = imresize(im,[250,250]);

end

end
