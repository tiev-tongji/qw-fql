from agents.dfq import DFQAgent
from agents.fql import FQLAgent
from agents.gfp import GFPAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.rebrac import ReBRACAgent
from agents.sac import SACAgent

agents = dict(
    dfq=DFQAgent,
    fql=FQLAgent,
    gfp=GFPAgent,
    ifql=IFQLAgent,
    iql=IQLAgent,
    rebrac=ReBRACAgent,
    sac=SACAgent,
)
