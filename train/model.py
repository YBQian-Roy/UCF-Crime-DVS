
import torch.nn.init as torch_init
from spikingjelly.clock_driven.neuron import MultiStepLIFNode,MultiStepIFNode
from spikingjelly.clock_driven import surrogate, neuron
from functools import partial
from layers import GraphConvolution, DistanceAdj,SimilarityAdj
from torch.nn.utils import weight_norm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)




class Model_single(torch.nn.Module):
    def __init__(self, n_feature):
        super(Model_single, self).__init__()
        self.fc = nn.Linear(n_feature,n_feature)
        self.classifier = nn.Linear(n_feature, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.7)
        self.apply(weights_init)





    def forward(self, inputs, is_training=True):
        x = self.fc(inputs)
        # x = inputs
        # if is_training:
        #     x = self.dropout(x)
        return x, self.sigmoid(self.classifier(x))





class Model_MSF(torch.nn.Module):
    def __init__(self, n_feature):
        super(Model_MSF, self).__init__()
        self.fc = nn.Linear(n_feature,n_feature)
        self.lif = MultiStepLIFNode(tau=1.6, detach_reset=True, backend='cupy')
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.8)
        self.classifier = nn.Linear(n_feature, 1)
        self.tim = TIM(TIM_alpha=0.6,in_channels=256)
        self.MSF = MSF(len_feature=256)

    def forward(self, x, is_training=True):
        x = self.MSF(x)
        if is_training:
            x = self.dropout(x)
        return x, self.sigmoid(self.classifier(x))





class TIM(BaseModule):
    def __init__(self, dim=256, encode_type='direct', in_channels=16, TIM_alpha=0.5):
        super().__init__(step=1, encode_type=encode_type)

        # self.interactor = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1,
        #                             padding=0, bias=True)
        self.interactor = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,
                  stride=1, padding=1, bias=False)
        self.in_lif = LIFNode(threshold=0.3,tau=1.6,detach_reset=True, backend='cupy')
        self.out_lif = LIFNode(threshold=0.5, tau=1.6, detach_reset=True, backend='cupy')
        self.tim_alpha = TIM_alpha

    def forward(self, x):
        B, T, F = x.shape
        x = x.permute(1, 2, 0).unsqueeze(-1)  # [T, F, 1, B]-
        output = []
        # x_tim = torch.empty_like(x[:, 0, :])
        x_tim = torch.empty_like(x[0])

        for i in range(T):
            if i == 0:
                # x_tim = x[:, i, :]
                # output.append(x_tim)
                x_tim = x[i]
                output.append(x_tim.squeeze(-1))
            else:
                # x_tim = self.interactor(x_tim.unsqueeze(1)).squeeze(1)
                # x_tim = self.in_lif(x_tim) * self.tim_alpha + x[:, i, :] * (1 - self.tim_alpha)
                # x_tim = self.out_lif(x_tim)
                # output.append(x_tim)
                x_tim = self.interactor(x_tim.flatten(1, 2)).reshape(F, B, -1).contiguous()
                x_tim = self.in_lif(x_tim) * self.tim_alpha + x[i] * (1 - self.tim_alpha)
                x_tim = self.out_lif(x_tim)
                output.append(x_tim.squeeze(-1))

        output = torch.stack(output, dim=0)  # B T F
        output = output.permute(2, 0, 1)
        return output


class MSF(nn.Module):
    def __init__(self, len_feature):
        super(MSF, self).__init__()
        bn = nn.BatchNorm1d
        self.dropout = nn.Dropout(0.8)

        self.len_feature = len_feature
        self.lif = MultiStepLIFNode(tau=1.6, detach_reset=True, backend='cupy')  # new code

        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=64, kernel_size=3,
                      stride=1,dilation=1, padding=1),
            bn(64),
            # MultiStepLIFNode(tau=1.5, detach_reset=True, backend='cupy')  # new code,
            # bn(64)
            # nn.dropout(0.7)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=64, kernel_size=3,
                      stride=1, dilation=2, padding=2),
            bn(64),
            # MultiStepLIFNode(tau=1.5, detach_reset=True, backend='cupy')
            # bn(64)
            # nn.dropout(0.7)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=64, kernel_size=3,
                      stride=1, dilation=4, padding=4),
            bn(64),
            # MultiStepLIFNode(tau=1.5, detach_reset=True, backend='cupy')
            # bn(64)
            # nn.dropout(0.7),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=64, kernel_size=1,
                      stride=1, padding=0, bias = False),
            # MultiStepLIFNode(tau=1.5, detach_reset=True, backend='cupy'),
            # nn.dropout(0.7),
        )
        self.SpikingGCN=SpikingGCN(64)
        self.tim = TIM(TIM_alpha=0.6,in_channels=256)


    def forward(self, x,is_Snn=False):
        # x: (B, T, F)
        out = x.permute(0, 2, 1)  # x:   B ,F,T
        # out = x.permute(1, 2, 0)  #(T,F,B)

        residual = out
        out1 = self.conv_1(out)
        out1 = self.lif(out1.permute(2, 0, 1)).permute(1, 2, 0)
        # out1 = self.lif(out1)

        out1 = self.dropout(out1)

        out2 = self.conv_2(out)
        out2 = self.lif(out2.permute(2, 0, 1)).permute(1, 2, 0)
        # out2 = self.lif(out2)

        out2 = self.dropout(out2)

        out3 = self.conv_3(out)
        out3 = self.lif(out3.permute(2, 0, 1)).permute(1, 2, 0)
        # out3 = self.lif(out3)

        out3 = self.dropout(out3)

        out_d = torch.cat((out1, out2, out3), dim=1)
        #
        #
        out = self.conv_4(out)
        out = self.lif(out.permute(2, 0, 1)).permute(1, 2, 0)
        # # # out = self.lif(out)
        # #
        out = self.dropout(out)
        # # #
        # # #
        out = self.SpikingGCN(out, None)
        out = self.lif(out.permute(2, 0, 1)).permute(1, 2, 0)

        out = torch.cat((out_d, out), dim=1)

        out = out.permute(0, 2, 1)
        # # # out = out.permute(2, 0, 1)
        out = self.tim(out)
        out = out.permute(0, 2, 1)
        # out = out.permute(1, 2, 0)

        out = out + residual

        # out = out + out4 + residual

        out = out.permute(0, 2, 1)
        # out = out.permute(2, 0, 1)
        # out: (B, T, 1)

        return out

class SpikingGCN(torch.nn.Module):
    def __init__(self,visual_width):
        super(SpikingGCN,self).__init__()
        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.lif = MultiStepLIFNode(tau=1.6, detach_reset=True, backend='cupy')
        #
        # self.linear_lif = nn.Sequential(
        #     nn.Linear(visual_width, visual_width),
        #     MultiStepLIFNode(tau=1.5, detach_reset=True, backend='cupy')
        # )




    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1))  # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2 / (x_norm_x + 1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output


    def forward(self,x,lengths):
        # x = x.permute(1, 0, 2)
        x = x.permute(0, 2, 1)


        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1])

        #if is_snn = true
        # x_test = self.gc1(x,adj)
        x1_h = self.lif(self.gc1(x, adj).permute(1,0,2)).permute(1,0,2)
        x2_h = self.lif(self.gc3(x, disadj).permute(1,0,2)).permute(1,0,2)

        x1 = self.lif(self.gc2(x1_h, adj).permute(1,0,2)).permute(1,0,2)
        x2 = self.lif(self.gc4(x2_h, disadj).permute(1,0,2)).permute(1,0,2)

        # if is_snn = false
        # x1_h = self.lif(self.gc1(x, adj))
        # x2_h = self.lif(self.gc3(x, disadj))
        #
        # x1 = self.lif(self.gc2(x1_h, adj))
        # x2 = self.lif(self.gc4(x2_h, disadj))

        x = torch.cat((x1, x2), 2)

        # x = x.permute(1, 0, 2)
        # x = self.linear_lif(x)
        # x = x.permute(1,0,2)
        x = self.linear(x)
        # x = self.lif(x).permute(1,2,0)

        x = x.permute(0, 2, 1)

        return x






def model_generater(model_name, feature_size):
    if model_name == 'model_single':
        model = Model_single(feature_size)  # for anomaly detection, only one class, anomaly, is needed.\
    elif model_name == 'MSF':
        model = Model_MSF(feature_size)
    else:
        raise ('model_name is out of option')
    return model