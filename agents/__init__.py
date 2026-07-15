from agents.qw_fql import QW_FQLAgent 
from agents.fql import FQLAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.rebrac import ReBRACAgent
from agents.sac import SACAgent

agents = dict(
    qw_fql=QW_FQLAgent,
    fql=FQLAgent,
    ifql=IFQLAgent,
    iql=IQLAgent,
    rebrac=ReBRACAgent,
    sac=SACAgent,
)
