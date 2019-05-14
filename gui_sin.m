function varargout = gui_sin(varargin)
% GUI_SIN MATLAB code for gui_sin.fig
%      GUI_SIN, by itself, creates a new GUI_SIN or raises the existing
%      singleton*.
%
%      H = GUI_SIN returns the handle to a new GUI_SIN or the handle to
%      the existing singleton*.
%
%      GUI_SIN('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GUI_SIN.M with the given input arguments.
%
%      GUI_SIN('Property','Value',...) creates a new GUI_SIN or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before gui_sin_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to gui_sin_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help gui_sin

% Last Modified by GUIDE v2.5 26-Apr-2019 12:03:39

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @gui_sin_OpeningFcn, ...
                   'gui_OutputFcn',  @gui_sin_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before gui_sin is made visible.
function gui_sin_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to gui_sin (see VARARGIN)

% Choose default command line output for gui_sin
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes gui_sin wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = gui_sin_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
   
dbPath = '/home/user/ElezCET/latest/test images';
dbPath = fullfile(dbPath,'*.jpg');
[fn,pn] = uigetfile(dbPath);
img1 = fullfile(pn,fn);
axes(handles.axes1);
imshow(img1);title('Query Image');
handles.ImgData1 = img1;
guidata(hObject,handles);
%fprintf('\nSelected image:\n\t:%s',filename);


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

clc;
set(handles.edit1,'string','');
I3 = handles.ImgData1;
fprintf('\nLoading models...')
load('net_alexnet.mat')
load('net_dleaf.mat')
load('net_vgg16.mat')
load('SVM.mat')
load('pca_dim.mat')
fprintf('\nPreprocessing...')
im1 = imPreprocessor1(I3);
im2 = imPreprocessor2(I3);
im3 = imPreprocessor3(I3);
dbPath1 = '/home/user/ElezCET/animal_database';
imds2 = imageDatastore(dbPath1,...
    'IncludeSubfolders', true,...
    'LabelSource','foldernames',...
    'ReadFcn',@imPreprocessor2);
[traindb2,testdb2] = splitEachLabel(imds2,0.7,'randomized');
fprintf('\tDone.')

fprintf('\nExtracting features...')

layer = 'drop7';
feat1 = activations(net1,im1,layer,'OutputAs','rows');
feat1 = feat1*reducedDimension;

layer = 'drop7';
feat2 = activations(net2,im2,layer,'OutputAs','rows');
feat2 = feat2*reducedDimension1;

layer = 'fc2';
feat3 = activations(net3,im3,layer,'OutputAs','rows');
feat3 = feat3*reducedDimension3;

fprintf('\tDone.')


FFV_test = (feat1+ feat2 + feat3) / 3;

y_test = testdb2.Labels;
Y_pred = predict(SVM_Model,FFV_test);
pred = upper(string(Y_pred));


%if accuracy1 < 50
 %   pred = 'UNABLE TO PREDICT';
  %  fprintf(' \nPrediction:\n\t UNABLE TO PREDICT');
   % fprintf('\nAccuracy:%f',accuracy1);
%else
    
    fprintf('\nPrediction:\t%s',pred);
    
%end

set(handles.edit1,'string',pred);



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --------------------------------------------------------------------
function Untitled_1_Callback(hObject, eventdata, handles)
% hObject    handle to Untitled_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
