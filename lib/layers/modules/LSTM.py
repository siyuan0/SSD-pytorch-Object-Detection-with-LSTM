import torch
import torch.nn as nn
import torch.nn.functional as F

CONTEXT = {'use LSTM': True}

class no_LSTM(object):
#acts as a wrapper to disable usage of LSTM, kinda like torch.no_grad()
    # def __init__(self):
    #     super(no_LSTM,self).__init__()
    def __enter__(self):
        self.prev = CONTEXT['use LSTM']
        CONTEXT['use LSTM'] = False
    def __exit__(self, *args):
        CONTEXT['use LSTM'] = self.prev

def reset_model_LSTM(model): 
#search through a model for LSTM layers and reset them
    reset_model = model
    for name, module in reset_model.named_children():
        try:
            module.reset_cell_state()
        except:
            reset_model_LSTM(module)


class LSTM_conv_dw(nn.Module):
# implementation of bottleneck-LSTM as described in http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Mobile_Video_Object_CVPR_2018_paper.pdf
# can also simply act as _conv_dw function
    def __init__(self, inp, oup, conv_dw_stride=1, conv_dw_padding=0, conv_dw_expand_ratio=1, use_LSTM=True, backprop_steps = 10):
        super(LSTM_conv_dw,self).__init__()
        self.use_LSTM = use_LSTM
        self._conv_dw = conv_dw(inp, oup, stride=conv_dw_stride, padding=conv_dw_padding, expand_ratio=conv_dw_expand_ratio)
        self.oup_channels = oup

        #to initialize cell state and pre_output during first time forward is called
        self.cell_state = None #should have shape: (batch_size, oup, (height+padding-2)/stride, (width+padding-2)/stride)
        self.pre_output = None #should have shape: (batch_size, oup, (height+padding-2)/stride, (width+padding-2)/stride)
        self.cell_initialized = False

        self.conv_bottleneck = nn.Conv2d(2*self.oup_channels, self.oup_channels, 1, 1, 0)
        self.conv_forgetgate = nn.Conv2d(self.oup_channels, self.oup_channels, 1, 1, 0)
        self.conv_inputgate = nn.Conv2d(self.oup_channels, self.oup_channels, 1, 1, 0)
        self.conv_input2cell = nn.Conv2d(self.oup_channels, self.oup_channels, 1, 1, 0)
        self.conv_outputgate = nn.Conv2d(self.oup_channels, self.oup_channels, 1, 1, 0)
        self.backprop_steps = backprop_steps


    def init_cell_state(self, sample_input):
    # initializes the cell_state and creates a dummy pre_output
    # Args: input:  tensor in shape of (batch_size, n_channels, height, width)
        self.batch_size = sample_input.size()[0]
        self.cell_state = [torch.empty(self.batch_size, self.oup_channels, 
                                        sample_input.size()[2], sample_input.size()[3])]
        self.pre_output = [torch.empty(self.batch_size, self.oup_channels, 
                                        sample_input.size()[2], sample_input.size()[3])]
        self.cell_initialized = True

    def reset_cell_state(self): 
    # resets the LSTM cell state. Cell state will automatically re-initialize during next forward
    # use .__class__.__name__ to find this layergit 
        self.batch_size = 0
        self.cell_state = None
        self.pre_output = None
        self.cell_initialized = False   

    def forward(self, x):
    # the option to turn off the LSTM portion is to allow for initial training as a non temporally-aware Neural Net
    # if use_LSTM=False, this class behaves the same as _conv_dw
    # Args: input:  tensor in shape of (batch_size, n_channels, height, width)
        if self.use_LSTM and CONTEXT['use LSTM']:
            #merging new info with previous output
            input = self._conv_dw(x)
            
            with torch.no_grad():
                #initializes cell state if not yet initialized
                if self.cell_initialized == False: self.init_cell_state(input) 

            while len(self.cell_state) > self.backprop_steps:
                #delete history that are older than specified
                del self.cell_state[0]
                del self.pre_output[0]

            pre_output = self.pre_output[-1].detach() #to seperate the entire history from autograd
            cell_state = self.cell_state[-1].detach()

            merged_inp = self.conv_bottleneck(torch.cat((input, pre_output),dim=1))
            
            #forgetting info from cell state based on new info
            forget_filter = F.sigmoid(self.conv_forgetgate(merged_inp))
            cell_state = torch.mul(cell_state, forget_filter)

            #adding new info to cell state
            input2cell = torch.mul(F.tanh(self.conv_input2cell(merged_inp)), F.sigmoid(self.conv_inputgate(merged_inp)))
            cell_state = cell_state + input2cell

            #using cell state to generate output
            output = F.tanh(torch.mul(cell_state, self.conv_outputgate(merged_inp)))
            pre_output = output

            self.cell_state.append(cell_state)
            self.pre_output.append(pre_output)

            return output

        else:
            output = self._conv_dw(x)
            self.pre_output = output
            return output

def conv_dw(inp, oup, stride=1, padding=0, expand_ratio=1):
# based on the implementation in https://github.com/tensorflow/models/blob/master/research/object_detection/models/feature_map_generators.py#L213
# when the expand_ratio is 1, the implemetation is nearly same. Since the shape is always change, I do not add the shortcut as what mobilenetv2 did.
    # input tensor: (batch_size, inp, height, width)
    # output tensor: (batch_size, oup, (height+padding-2)/stride, (width+padding-2)/stride)
    return nn.Sequential(
        # pw
        nn.Conv2d(inp, oup * expand_ratio, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup * expand_ratio),
        nn.ReLU6(inplace=True),
        # dw
        nn.Conv2d(oup * expand_ratio, oup * expand_ratio, 3, stride, padding, groups=oup * expand_ratio, bias=False),
        nn.BatchNorm2d(oup * expand_ratio),
        nn.ReLU6(inplace=True),
        # pw-linear
        nn.Conv2d(oup * expand_ratio, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )