from . import flownet1s
from . import flownet1s_irr
from . import flownet1s_irr_bi
from . import flownet1s_irr_occ
from . import flownet1s_irr_occ_bi
from . import IRR_FlowNet

from . import pwcnet
from . import pwcnetRecordable
from . import pwcnetFusion
from . import pwcnetX1Zero
from . import pwcnetConvConnector
from . import pwcnetWOX1Connection
from . import pwcExpertNet
from . import pwcnet_bi
from . import pwcnet_occ
from . import pwcnet_occ_bi
from . import pwcnet_irr
from . import pwcnet_irr_bi
from . import pwcnet_irr_occ
from . import pwcnet_irr_occ_bi
from . import IRR_PWC


FlowNet1S            = flownet1s.FlowNet1S
FlowNet1S_irr        = flownet1s_irr.FlowNet1S
FlowNet1S_irr_bi     = flownet1s_irr_bi.FlowNet1S
FlowNet1S_irr_occ    = flownet1s_irr_occ.FlowNet1S
FlowNet1S_irr_occ_bi = flownet1s_irr_occ_bi.FlowNet1S

PWCNet               = pwcnet.PWCNet
PWCNetRecordable     = pwcnetRecordable.PWCNetRecordable
PWCNetFusion         = pwcnetFusion.PWCNetFusion
PWCNetX1Zero         = pwcnetX1Zero.PWCNetX1Zero
PWCNetConv13Fusion     = pwcnetFusion.PWCNetConv13Fusion  # 3 conv layers - kernel size 1
PWCNetConv33Fusion     = pwcnetFusion.PWCNetConv33Fusion  # 3 conv layers - kernel size 3
PWCTrainableConvConnector11 = pwcnetConvConnector.PWCTrainableConvConnector11  # 1 conv layer - kernel size 1
PWCTrainableConvConnector12 = pwcnetConvConnector.PWCTrainableConvConnector12  # 2 conv layers - kernel size 1
PWCTrainableConvConnector13 = pwcnetConvConnector.PWCTrainableConvConnector13  # 3 conv layers - kernel size 1
PWCTrainableConvConnector31 = pwcnetConvConnector.PWCTrainableConvConnector31  # 1 conv layer - kernel size 3
PWCTrainableConvConnector32 = pwcnetConvConnector.PWCTrainableConvConnector32  # 2 conv layers - kernel size 3
PWCTrainableConvConnector33 = pwcnetConvConnector.PWCTrainableConvConnector33  # 3 conv layers - kernel size 3
PWCAppliedConvConnector13 = pwcnetConvConnector.PWCAppliedConvConnector13
PWCAppliedConvConnector33 = pwcnetConvConnector.PWCAppliedConvConnector33
PWCNetLinCombFusion  = pwcnetFusion.PWCNetLinCombFusion
PWCConnector1        = pwcnetConvConnector.PWCConvConnector1
PWCConnector3        = pwcnetConvConnector.PWCConvConnector3
PWCConvAppliedConnector = pwcnetConvConnector.PWCConvAppliedConnector
PWCLinCombAppliedConnector = pwcnetConvConnector.PWCLinCombAppliedConnector
PWCNetWOX1Connection = pwcnetWOX1Connection.PWCNetWOX1Connection
PWCExpertNet         = pwcExpertNet.PWCExpertNet
PWCNet_bi            = pwcnet_bi.PWCNet
PWCNet_occ           = pwcnet_occ.PWCNet
PWCNet_occ_bi        = pwcnet_occ_bi.PWCNet
PWCNet_irr           = pwcnet_irr.PWCNet
PWCNet_irr_bi        = pwcnet_irr_bi.PWCNet
PWCNet_irr_occ       = pwcnet_irr_occ.PWCNet
PWCNet_irr_occ_bi    = pwcnet_irr_occ_bi.PWCNet

IRR_FlowNet          = IRR_FlowNet.FlowNet1S
IRR_PWC              = IRR_PWC.PWCNet

