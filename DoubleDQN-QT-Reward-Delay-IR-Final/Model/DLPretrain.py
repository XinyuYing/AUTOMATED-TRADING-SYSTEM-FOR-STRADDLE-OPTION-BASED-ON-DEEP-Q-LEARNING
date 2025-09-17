from BackUp.Guidance import Guidance
from Model.Env import Env
from Setting import arg
#生成数据，进行深度学习

if __name__ == '__main__':
    env = Env(data_path="Data/15m000300/RESSET_INDXSH2022_000300.csv")
    guidance=Guidance(env)

    while env.TimeCursor<env.DataLen-arg.ADayTime-1:
        action=guidance.getGuidanceAction()
        print(action)
        env.step(action)