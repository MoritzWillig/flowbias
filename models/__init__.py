from . import flownet1s

from . import pwcnet
from . import pwcnetFusion
from . import pwcnetX1Zero
from . import pwcnetConvConnector
from . import pwcnetWOX1Connection
from . import pwcnetWOX1ConnectionExt
from . import pwcnetDSEncoder
from . import pwcExpertNet
from . import pwcExpertAddNet
from . import pwcExpertLinAddNet
from . import pwcExpertNetWOX1
from . import pwcExpertLinAddNetWOX1
from . import pwcExpertAddNetWOX1
from . import pwcnetRecordable
from . import pwcnetWOX1ConnectionRecordable
from . import pwcExpertAddNetRecordable

from . import pwcnet_residual_flow
from . import pwcnet_secondary_flow


FlowNet1S            = flownet1s.FlowNet1S

PWCNet               = pwcnet.PWCNet
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
PWCNetWOX1ConnectionExt = pwcnetWOX1ConnectionExt.PWCNetWOX1ConnectionExt
PWCNetDSEncoder = pwcnetDSEncoder.PWCNetDSEncoder
PWCExpertNet         = pwcExpertNet.PWCExpertNet
CTSKPWCExpertNet02   = pwcExpertNet.CTSKPWCExpertNet02
PWCExpertAddNet      = pwcExpertAddNet.PWCExpertAddNet
CTSKPWCExpertNetAdd01   = pwcExpertAddNet.CTSKPWCExpertNetAdd01
CTSPWCExpertNetAdd01 = pwcExpertAddNet.CTSPWCExpertNetAdd01
PWCExpertLinAddNet      = pwcExpertLinAddNet.PWCExpertLinAddNet
CTSKPWCExpertNetLinAdd01   = pwcExpertLinAddNet.CTSKPWCExpertNetLinAdd01
CTSPWCExpertNetLinAdd01 = pwcExpertLinAddNet.CTSPWCExpertNetLinAdd01
PWCExpertAddNetWOX1      = pwcExpertAddNetWOX1.PWCExpertAddNetWOX1
CTSKPWCExpertNetWOX1Add01   = pwcExpertAddNetWOX1.CTSKPWCExpertNetWOX1Add01
CTSKPWCExpertNet01WOX1Add   = pwcExpertAddNetWOX1.CTSKPWCExpertNetWOX1Add01
CTSPWCExpertNetWOX1Add01   = pwcExpertAddNetWOX1.CTSPWCExpertNetWOX1Add01
PWCExpertLinAddNetWOX1      = pwcExpertLinAddNetWOX1.PWCExpertLinAddNetWOX1
CTSKPWCExpertNetWOX1LinAdd01   = pwcExpertLinAddNetWOX1.CTSKPWCExpertNetWOX1LinAdd01
CTSPWCExpertNetWOX1LinAdd01   = pwcExpertLinAddNetWOX1.CTSPWCExpertNetWOX1LinAdd01
PWCExpertNetWOX1      = pwcExpertNetWOX1.PWCExpertNetWOX1
CTSKPWCExpertNet02WOX1   = pwcExpertNetWOX1.CTSKPWCExpertNet02WOX1
CTSPWCExpertNet02WOX1   = pwcExpertNetWOX1.CTSPWCExpertNet02WOX1
PWCNetRecordable     = pwcnetRecordable.PWCNetRecordable
PWCNetWOX1ConnectionRecordable = pwcnetWOX1ConnectionRecordable.PWCNetWOX1ConnectionRecordable
PWCExpertAddNetRecordable = pwcExpertAddNetRecordable.PWCExpertAddNetRecordable
CTSKPWCExpertNetAdd01Recordable = pwcExpertAddNetRecordable.CTSKPWCExpertNetAdd01Recordable
CTSPWCExpertNetAdd01Recordable = pwcExpertAddNetRecordable.CTSPWCExpertNetAdd01Recordable

PWCNetResidualFlow = pwcnet_residual_flow.PWCNetResidualFlow

CTSKPWCExpertNetWOX1Add01EncoderExpertsOnly = pwcExpertAddNetWOX1.CTSKPWCExpertNetWOX1Add01EncoderExpertsOnly
CTSKPWCExpertNetWOX1Add01DecoderExpertsOnly = pwcExpertAddNetWOX1.CTSKPWCExpertNetWOX1Add01DecoderExpertsOnly

PWCNetWOX1SecondaryFlow = pwcnet_secondary_flow.PWCNetWOX1SecondaryFlow
