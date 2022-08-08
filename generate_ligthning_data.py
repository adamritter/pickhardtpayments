import json
import timeit
def readlightninggraph(fn):
    f=open(fn)
    j=json.load(f)
    f.close()
    channels=j["channels"]
    n=0
    ids={}
    def getid(s):
        nonlocal n, ids
        if s not in ids:
            ids[s]=n
            n+=1
        return ids[s]
    g=[[getid(channel["source"]), getid(channel["destination"]), channel["satoshis"], channel["fee_per_millionth"]] 
            for channel in channels if channel["base_fee_millisatoshi"]==0]
    return [g, ids]

start = timeit.default_timer()
lightning_edges, ids=readlightninggraph("examples/listchannels20220412.json")
end = timeit.default_timer()
def num_nodes(edges):
    nodes=set()
    r=0
    for edge in edges:
        if edge[0] not in nodes:
            nodes.add(edge[0])
            r+=1
        if edge[1] not in nodes:
            nodes.add(edge[1])
            r+=1
    return r
print(len(lightning_edges), num_nodes(lightning_edges))
rene = ids["03efccf2c383d7bf340da9a3f02e2c23104a0e4fe8ac1a880c8e2dc92fbdacd9df"]
otto = ids["027ce055380348d7812d2ae7745701c9f93e70c1adeb2657f053f91df4f2843c71"]
tested_amount = 37077242 #37 million sats
# write simpler file
f=open("lightning.data", "w")
log_probability_cost_multiplier=10000000  # 100M / 150M doesn't work well
print(num_nodes(lightning_edges), len(lightning_edges), rene, otto,
    tested_amount, log_probability_cost_multiplier, file=f)
for lightning_edge in lightning_edges:
    # Guaranteed liquidity is 0 in test file.
    print(lightning_edge[0], lightning_edge[1], lightning_edge[2], lightning_edge[3], 0, file=f)
f.close()
print("written")
