import torch.nn as nn
from gnn_conv import GCNConv, GINConv, GATConv

class GCN(nn.Module):
    def __init__(self, args, in_f, hid, out_f):
        super().__init__(); self.c1=GCNConv(args,in_f,hid,"conv1"); self.c2=GCNConv(args,hid,out_f,"conv2")
    def forward(self,x,A,args): return self.c2(self.c1(x,A,args).relu(),A,args)
class GIN(nn.Module):
    def __init__(self, args, in_f, hid, out_f):
        super().__init__(); self.c1=GINConv(args,in_f,hid,"conv1"); self.c2=GINConv(args,hid,out_f,"conv2")
    def forward(self,x,A,args): return self.c2(self.c1(x,A,args).relu(),A,args)
class GAT(nn.Module):
    def __init__(self, args, in_f, hid, out_f):
        super().__init__(); self.c1=GATConv(args,in_f,hid,heads=1,layer_name="conv1"); self.c2=GATConv(args,hid,out_f,heads=1,layer_name="conv2")
    def forward(self,x,A,args): return self.c2(self.c1(x,A,args).relu(),A,args)

############################################################################