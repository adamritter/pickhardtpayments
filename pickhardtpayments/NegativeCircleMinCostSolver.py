import os
class NegativeCircleMinCostSolver:
    OPTIMAL=1
    def Solve(self):
        os.system("../min_cost")
        # os.system("../target/release/min_cost_rs")
        f=open("min_cost.out", "r")
        self.flows=[]
        for i in range(int(f.readline())):
            self.flows.append(int(f.readline()))
        f.close()
        return self.OPTIMAL
    def NumArcs(self):
        return len(self.flows)
    def Flow(self, i):
        return self.flows[i]