// TODO: add back 0 cost edges for negative cycle, make it work (now buggy :( ))
// min_cost alternative implementation
// Usage: g++ min_cost.cc -std=c++17 -O2 -lm -o min_cost && ./min_cost

#include <algorithm>
#include <chrono>
#include <iostream>
#include <list>
#include <math.h>
#include <stdio.h>
#include <tuple>
#include <vector>

using namespace std;

struct OriginalEdge {
    int u, v, capacity, cost, flow, guaranteed_liquidity;
};

std::chrono::steady_clock::time_point now() {
    return std::chrono::steady_clock::now();
}
void elapsed(const char*s, std::chrono::steady_clock::time_point begin) {
    std::cout << "Time difference for " << s << " = " << std::chrono::duration_cast<std::chrono::milliseconds>(now() - begin).count() << "ms" << std::endl;

}

// https://www.geeksforgeeks.org/dinics-algorithm-maximum-flow/
// C++ implementation of Dinic's Algorithm for Maximum Flow
using namespace std;
 
// A structure to represent a edge between
// two vertex
struct MaxFlowEdge {
    int v; // Vertex v (or "to" vertex)
           // of a directed edge u-v. "From"
           // vertex u can be obtained using
           // index in adjacent array.
 
    int flow; // flow of data in edge
 
    int C; // capacity
 
    int rev; // To store index of reverse
             // edge in adjacency list so that
             // we can quickly find it.
};
 
// Residual MaxFlowGraph
class MaxFlowGraph {
    int V; // number of vertex
    int* level; // stores level of a node
 
public:
    vector<MaxFlowEdge>* adj;

    MaxFlowGraph(int V)
    {
        adj = new vector<MaxFlowEdge>[V];
        this->V = V;
        level = new int[V];
    }
    ~MaxFlowGraph() {
        delete[] adj;
        delete[] level;
    }
 
    // add edge to the graph
    int addEdge(int u, int v, int C)
    {
        // Forward edge : 0 flow and C capacity
        MaxFlowEdge a{ v, 0, C, (int)adj[v].size() };
 
        // Back edge : 0 flow and 0 capacity
        MaxFlowEdge b{ u, 0, 0, (int)adj[u].size() };
 
        adj[u].push_back(a);
        adj[v].push_back(b); // reverse edge
        return adj[u].size()-1;
    }
 
    bool BFS(int s, int t);
    int sendFlow(int s, int flow, int t, int ptr[]);
    int DinicMaxflow(int s, int t, int limit);
};
 
// Finds if more flow can be sent from s to t.
// Also assigns levels to nodes.
bool MaxFlowGraph::BFS(int s, int t)
{
    for (int i = 0; i < V; i++)
        level[i] = -1;
 
    level[s] = 0; // Level of source vertex
 
    // Create a queue, enqueue source vertex
    // and mark source vertex as visited here
    // level[] array works as visited array also.
    list<int> q;
    q.push_back(s);
 
    vector<MaxFlowEdge>::iterator i;
    while (!q.empty()) {
        int u = q.front();
        q.pop_front();
        for (i = adj[u].begin(); i != adj[u].end(); i++) {
            MaxFlowEdge& e = *i;
            if (level[e.v] < 0 && e.flow < e.C) {
                // Level of current vertex is,
                // level of parent + 1
                level[e.v] = level[u] + 1;
 
                q.push_back(e.v);
            }
        }
    }
 
    // IF we can not reach to the sink we
    // return false else true
    return level[t] < 0 ? false : true;
}
 
// A DFS based function to send flow after BFS has
// figured out that there is a possible flow and
// constructed levels. This function called multiple
// times for a single call of BFS.
// flow : Current flow send by parent function call
// start[] : To keep track of next edge to be explored.
//           start[i] stores  count of edges explored
//           from i.
//  u : Current vertex
//  t : Sink
int MaxFlowGraph::sendFlow(int u, int flow, int t, int start[])
{
    // Sink reached
    if (u == t)
        return flow;
 
    // Traverse all adjacent edges one -by - one.
    for (; start[u] < adj[u].size(); start[u]++) {
        // Pick next edge from adjacency list of u
        MaxFlowEdge& e = adj[u][start[u]];
 
        if (level[e.v] == level[u] + 1 && e.flow < e.C) {
            // find minimum flow from u to t
            int curr_flow = min(flow, e.C - e.flow);
 
            int temp_flow
                = sendFlow(e.v, curr_flow, t, start);
 
            // flow is greater than zero
            if (temp_flow > 0) {
                // add flow  to current edge
                e.flow += temp_flow;
 
                // subtract flow from reverse edge
                // of current edge
                adj[e.v][e.rev].flow -= temp_flow;
                return temp_flow;
            }
        }
    }
 
    return 0;
}
 
// Returns maximum flow in graph
int MaxFlowGraph::DinicMaxflow(int s, int t, int limit)
{
    // Corner case
    if (s == t)
        return -1;
 
    int total = 0; // Initialize result
 
    // Augment the flow while there is path
    // from source to sink
    while (total < limit && BFS(s, t) == true) {
        // store how many edges are visited
        // from V { 0 to V }
        int* start = new int[V + 1]{ 0 };
 
        // while flow is not zero in graph from S to D
        while (int flow = sendFlow(s, limit-total, t, start)) {
 
            // Add path flow to overall flow
            total += flow;
        }
    }
 
    // return maximum flow
    return total;
}


// Using the Shortest Path Faster Algorithm to find a negative cycle
// https://konaeakira.github.io/posts/using-the-shortest-path-faster-algorithm-to-find-negative-cycles.html
#include <queue>
const long long INF = std::numeric_limits<long long>::max() / 4;

std::vector<int> inline detect_cycle(int n, int* pre)
{
    bool visited[n], on_stack[n];
    std::vector<int> vec;
    std::fill(on_stack, on_stack + n, false);
    std::fill(visited, visited + n, false);
    for (int i = 0; i < n; ++i)
        if (!visited[i])
        {
            for (int j = i; j != -1; j = pre[j])
                if (!visited[j])
                {
                    visited[j] = true;
                    vec.push_back(j);
                    on_stack[j] = true;
                }
                else
                {
                    if (on_stack[j]) {
                        int jj=0;
                        while(vec[jj]!=j)
                            jj++;
                        vector<int> vec2=vector(vec.begin()+jj, vec.end());
                        reverse(vec2.begin(), vec2.end());
                        return vec2;
                    }
                    break;
                }
            for (int j : vec)
                on_stack[j] = false;
            vec.clear();
        }
    return vec;
}
#include <deque>
long long sum(deque<int>&q, long long* dis) {
    long long s=0;
    for(auto i=q.begin(); i!=q.end(); i++)
        s+=dis[*i];
    return s;
}

struct MinCostEdge {
    int v;
    int remaining_capacity;
    int cost;
    int reverse_idx;
    int guaranteed_liquidity;
};

vector<int> inline update_dis(int u, int v, int w, long long disu, int * pre, long long * dis,bool*in_queue,
    std::deque<int>& queue, int&iter, int n) {
                pre[v] = u;
				dis[v] = disu + w;
				if (++iter == n)
                {
                    iter = 0;
                    vector<int> cycle=detect_cycle(n, pre);
                    if (!cycle.empty()) {
                        return cycle;
                    }
                }
				if (!in_queue[v])
				{
                    int front=queue.front();
                    queue.push_back(v);
					in_queue[v] = true;
				}
                return vector<int>();
    }

vector<int> spfa_early_terminate(int n, std::vector<std::pair<int, int>> *adj, std::vector<MinCostEdge> *adj2)
{
    int pre[n];
    long long dis[n];
    bool in_queue[n];

	std::fill(dis, dis + n, 0);
	std::fill(pre, pre + n, -1);
	std::fill(in_queue, in_queue + n, true);
	std::deque<int> queue;
	for (int i = 0; i < n; ++i)
		queue.push_back(i);
    int iter = 0;
    int ct1=0, ct2=0;

	while (!queue.empty())
	{
		int u = queue.front();
		queue.pop_front();
		in_queue[u] = false;
        long long disu=dis[u];
        // cout << adj[u].size() << endl;
        
		for (int i=0; i<adj[u].size(); i++) {
            auto [v, w]=adj[u][i];
            auto disv=dis[v];
            if(pre[u]==v) {  // Don't allow cycles of 2.
                continue;
            }
            if (disu + w < disv)
			{
				vector<int> cycle=update_dis(u, v, w, disu,  pre, dis, in_queue, queue, iter, n);
                if(!cycle.empty()) { return cycle;}
			}
        }
	}
    return detect_cycle(n, pre);
}

long long total_cost(vector<OriginalEdge> & lightning_data) {
    long long r=0;
    for(int i=0; i<lightning_data.size(); i++) {
        auto edge=lightning_data[i];
        r+=((long long)edge.cost)*edge.flow;
        if(edge.cost < 0) {
            printf("negative cost!!!!!!\n");
        }
        if(edge.flow < 0) {
            printf("negative flow!!!!!!\n");
        }

        if(edge.flow > edge.capacity) {
            printf("overflow!!!!!!\n");
        }
    }
    return r;
}


long long adj_total_cost(int N, std::vector<MinCostEdge> *adj2) {
    long long total=0;
    for(int i=0; i<N; i++) {
        for(int j=0; j<adj2[i].size(); j++) {
            MinCostEdge e=adj2[i][j];
            if(e.cost < 0) {
                total-=((long long)e.cost)*e.remaining_capacity;
            }
        }
    }
    return total;
}



const bool use_guaranteed_capacity=true;

// Returns positive number
float minus_log_probability(MinCostEdge e, MinCostEdge er) {
    int from_total=(e.cost>0) ? e.remaining_capacity : er.remaining_capacity;
    int capacity=e.remaining_capacity+er.remaining_capacity;
    if(use_guaranteed_capacity && from_total+e.guaranteed_liquidity>=capacity) {
        return 0.0;
    }
    float p=(float(from_total)+1)/(capacity-e.guaranteed_liquidity+1);
    return -log2(p);  
}

long long adj_total_mlog_prob(int N, std::vector<MinCostEdge> *adj2) {
    double mlogp=0.0;
    for(int i=0; i<N; i++) {
        for(int j=0; j<adj2[i].size(); j++) {
            MinCostEdge e=adj2[i][j];
            if(e.cost < 0) {
                MinCostEdge er=adj2[e.v][e.reverse_idx];
                mlogp+=minus_log_probability(e, er);
            }
        }
    }
    return mlogp;
}

// Returns 1/(from_total+1)/log(2), the derivative of minus log probability
float dminus_log_probability(MinCostEdge e, MinCostEdge er) {
    const float log2inv=1.0/log(2.0);
    int from_total=(e.cost>0) ? e.remaining_capacity : er.remaining_capacity;
    // float invp=(e.remaining_capacity+er.remaining_capacity+1)/(from_total+1)*log2inv;
    // return invp;
    int capacity=e.remaining_capacity+er.remaining_capacity;
    if(use_guaranteed_capacity && from_total+e.guaranteed_liquidity>=capacity) {
        return 0.0;
    }
    return 1.0/float(from_total)/log2inv;
}

pair<int,int> getAdj(MinCostEdge e, MinCostEdge er, float log_probability_cost_multiplier) {
    if(e.remaining_capacity==0) {
        return make_pair(e.v, INT32_MAX/2);
    }
    int cost = e.cost;
    if(log_probability_cost_multiplier >= 0) {
        MinCostEdge e2=e, er2=er;
        e2.remaining_capacity-=1;
        er2.remaining_capacity+=1;
        cost+=round(log_probability_cost_multiplier*(minus_log_probability(e2, er2)));
        cost-=round(log_probability_cost_multiplier*(minus_log_probability(e, er)));
    }
    return make_pair(e.v, cost);
}

long long relative_cost_at(int at, vector<pair<MinCostEdge,MinCostEdge>> &edges, float log_probability_cost_multiplier) {
        long long r=0;
        for(int i=0; i<edges.size(); i++) {
            MinCostEdge e=edges[i].first;
            MinCostEdge er=edges[i].second;
            r+=((long long)e.cost)*at;
            MinCostEdge e2=e;
            MinCostEdge er2=er;
            e2.remaining_capacity-=at;
            er2.remaining_capacity+=at;
            r+=round(log_probability_cost_multiplier*(minus_log_probability(e2, er2)));
            r-=round(log_probability_cost_multiplier*(minus_log_probability(e, er)));
        }
        return r;
}

#ifdef DEBUG
    const bool debug=true;
#else
    const bool debug=false;
#endif

long long derivative_at(int at, vector<pair<MinCostEdge,MinCostEdge>> &edges, float log_probability_cost_multiplier) {
    long long r=0;
    if(debug)
    cout << "at: " << at << endl;
    for(int i=0; i<edges.size(); i++) {
        MinCostEdge e=edges[i].first;
        MinCostEdge er=edges[i].second;
        e.remaining_capacity-=at;
        er.remaining_capacity+=at;
        if(debug)
        cout << "derivative cost=" << e.cost << ", remaining capacity: " << e.remaining_capacity
            << " reverse capacity " << er.remaining_capacity << ", calculated cost: "
            << getAdj(e, er, log_probability_cost_multiplier).second 
            << ", calculated simple cost: " << getAdj(e, er, 0.0).second 
            << endl;
        r+=getAdj(e, er, log_probability_cost_multiplier).second;
    }
    return r;
}


// Returns a non-negative local minima on a negative cycle. Returns a number between 0 and min_capacity.
// Derivative at 0 should be negative, relative cost at 0.1 negative as well.
// Derivative at min_capacity is probably close to infinity just like negative cost.
// Local minima has negative relative cost with 0 derivative.
// Algorithm: find 0 derivative using halving. If relative cost is positive, find 0 relative cost.
// If derivative is positive, restart. If derivative is negative, go back a bit and find 0 relative cost again.

void print_at(int at, vector<pair<MinCostEdge,MinCostEdge>> &edges, float log_probability_cost_multiplier) {
    cout << "  at " << at << " derivative: " << derivative_at(at, edges, log_probability_cost_multiplier)
        << ", relative cost: " << relative_cost_at(at, edges, log_probability_cost_multiplier) << endl; 
}

int find_local_minima(vector<pair<MinCostEdge,MinCostEdge>> &edges, float log_probability_cost_multiplier,
    int min_capacity) {
    int min_capacity0=min_capacity;
    bool debug=false;
    if(debug) {
        cout << "Find local minima called with " << edges.size() << " edges and log_probability_cost_multiplier="
            << log_probability_cost_multiplier << ", min_capacity=" << min_capacity << endl;
        for(int i=0; i<edges.size(); i++) {
            cout << "  fee:" << edges[i].first.cost << ", remaining capacity: "
                <<edges[i].first.remaining_capacity  << ", edge capacity:"
                << edges[i].first.remaining_capacity+edges[i].second.remaining_capacity
                << ", getAdj cost=" << getAdj(edges[i].first, edges[i].second, log_probability_cost_multiplier).second
                << endl;
        }
        print_at(0, edges, log_probability_cost_multiplier);
        print_at(1, edges, log_probability_cost_multiplier);
        print_at(5, edges, log_probability_cost_multiplier);
        print_at(min_capacity, edges, log_probability_cost_multiplier);
    }
    if(derivative_at(0, edges, log_probability_cost_multiplier) >= 0) {
        cout << "Not negative cycle!!!!!!";
        return 0;
    }
    if(derivative_at(min_capacity, edges, log_probability_cost_multiplier) <=0) {
        cout << "Not positive at min_capacity!!!!!!";
        return 0;
    }
    int upper=min_capacity;
    while(true) {
        int lower=0;
        // Positive derivative, find 0 or negative derivative where upper+1 is positive.
        while(upper>lower) {
            int mid=(lower+upper)/2;
            // cout << "mid " << mid << endl;
            if(derivative_at(mid, edges, log_probability_cost_multiplier) <= 0) {
                lower=mid;
                if(upper==lower+1) {
                    upper--;
                }
            } else {
                upper=mid-1;
            }
        }
        if(debug) {
            cout << " step 1: "; print_at(upper, edges, log_probability_cost_multiplier);
            print_at(upper-50, edges, log_probability_cost_multiplier);
            print_at(1, edges, log_probability_cost_multiplier);
        }
        while(upper < min_capacity0 &&
             relative_cost_at(upper, edges, log_probability_cost_multiplier) >=
             relative_cost_at(upper+1, edges, log_probability_cost_multiplier)){
                upper++;
        }
        if(upper<=0) {
            cout << "Why returning 0???" << endl;
            return 0;
        }
        if(relative_cost_at(upper, edges, log_probability_cost_multiplier) < 0) {
            return upper;
        }
        // Nonnegative relative cost with nonnegative derivative, find negative relative cost with upper+1 nonnegative.
        while(true) {
            lower=0;
            while(upper>lower) {
                int mid=(lower+upper)/2;
                // cout << "mid2: " << mid << " "<<relative_cost_at(mid, edges, log_probability_cost_multiplier) << endl;
                if(relative_cost_at(mid, edges, log_probability_cost_multiplier) < 0) {
                    lower=mid;
                    if(upper==lower+1) {
                        upper--;
                    }
                } else {
                    upper=mid-1;
                }
            }
            if(debug) {
                cout << " step 2: "; print_at(upper, edges, log_probability_cost_multiplier);
                print_at(upper+1, edges, log_probability_cost_multiplier);
                print_at(upper+2, edges, log_probability_cost_multiplier);
            }
            while(upper>=0 and derivative_at(upper, edges, log_probability_cost_multiplier) == 0) {
                upper--;
            }
            if(upper<=0) {
                while(upper < min_capacity0 &&
                    relative_cost_at(upper, edges, log_probability_cost_multiplier) >=
                    relative_cost_at(upper+1, edges, log_probability_cost_multiplier)){
                        upper++;
                }
                return upper;
            }
            if(derivative_at(upper, edges, log_probability_cost_multiplier) > 0) {
                break;
            }
            // negative derivative while negative relative cost found.
            upper-=1;  // There should be nonnegative relative cost again.
            if(relative_cost_at(upper, edges, log_probability_cost_multiplier) < 0) {
                    cout << "Error: relative cost should be positive" << endl;
                    return 0;
            }
        } // Total cost after optimizations: 0.137132%, p=2.91038e-09%
        // Positive derivative, start process again.
    }
}

// TODO: decrease min_capacity by finding 0 derivative???
bool decrease_total_cost(int N, std::vector<std::pair<int, int>> *adj, std::vector<MinCostEdge> *adj2,
    float log_probability_cost_multiplier) {
    // Find negative cycle
    
    auto begin=now();
    vector<int> negative_cycle=spfa_early_terminate(N, adj, adj2);
    // elapsed("early terminate negative_cycle", begin);
    begin=now();
    if(debug)
	    cout << "early terminate negative_cycle: " << negative_cycle.size() << endl;
    vector<int> min_costs;
    int min_capacity=INT32_MAX;
    vector<int> min_cost_idxs;
    vector<pair<MinCostEdge,MinCostEdge> > edges;

    if(debug) cout << "Possible edges:" << endl;
    for(int i=0; i<negative_cycle.size(); i++) {
        int u=negative_cycle[i], v=negative_cycle[(i+1)%negative_cycle.size()];
        if(debug) cout << "  " << u << "->" << v << ": costs=";
        auto edges_from = adj[u];
        int min_cost=INT32_MAX;
        int min_cost_idx=-1;
        for (int j=0; j<edges_from.size(); j++) {
            if(edges_from[j].first==v) {
                if(debug)
                cout << edges_from[j].second << "; ";
                if(edges_from[j].second < min_cost) {
                    min_cost=edges_from[j].second;
                    min_cost_idx=j;
                }
            }
        }
        if(debug)
        cout << endl;
        if(min_cost_idx==-1) {
            printf("min_cost_idx==-1!!!!!");
            return false;
        }
        if(adj2[u].size()<=min_cost_idx) {
            cout << "Bad index!!!!!" << adj[u].size() << ", " << adj2[u].size() << ", " <<  min_cost_idx << endl;
            return false;
        }
        MinCostEdge e=adj2[u][min_cost_idx];
        if(e.remaining_capacity < min_capacity) {
            min_capacity = e.remaining_capacity;
        }
        min_cost_idxs.push_back(min_cost_idx);
        edges.emplace_back(e, adj2[e.v][e.reverse_idx]);
    }
    if(min_capacity==0 || min_cost_idxs.size()==0) {
        return false;
    }
    if(log_probability_cost_multiplier >= 0) {
        if(debug)
        cout << "Derivative at 0: " << derivative_at(0, edges, log_probability_cost_multiplier)
            << ", relative cost at 0: " << relative_cost_at(0, edges, log_probability_cost_multiplier)
            << ", relative cost at 1: " << relative_cost_at(1, edges, log_probability_cost_multiplier)
            << ", derivative at " << min_capacity << ": " 
            << derivative_at(min_capacity, edges, log_probability_cost_multiplier)
            << endl;
        min_capacity=find_local_minima(edges, log_probability_cost_multiplier, min_capacity);
        // min_capacity=(relative_cost_at(floor(fmin_capacity), edges, log_probability_cost_multiplier) <
                        // relative_cost_at(floor(fmin_capacity)+1, edges, log_probability_cost_multiplier))
                        // ? floor(fmin_capacity) : (floor(fmin_capacity)+1);
        if(debug)
         cout << "Find local minima returned " << min_capacity << 
            ", derivative at 0: " << derivative_at(0, edges, log_probability_cost_multiplier)
            << ", derivative at new min capacity(" << min_capacity << "): " 
            << derivative_at(min_capacity, edges, log_probability_cost_multiplier)
            << ", relative cost at min_capacity: " <<
            relative_cost_at(min_capacity, edges, log_probability_cost_multiplier)
            << endl;
    }
    if(min_capacity==0 || min_cost_idxs.size()==0) {
        return false;
    }
    // printf("min capacity=%d\n", min_capacity);
    // decrease using min capacity
    
    if(debug) cout << "adjusted cost before modification: " << adj_total_cost(N, adj2) << "+" <<
        adj_total_mlog_prob(N, adj2) << "*" << log_probability_cost_multiplier << "=" <<
        adj_total_cost(N, adj2)+adj_total_mlog_prob(N, adj2)*log_probability_cost_multiplier
         << endl;

    for(int i=0; i<min_cost_idxs.size(); i++) {
        int u=negative_cycle[i];
        if(adj2[u].size()<=min_cost_idxs[i]) {
            cout << "Bad index2!!!!!";
            return false;
        }
        MinCostEdge *e=&adj2[u][min_cost_idxs[i]];
        int v=e->v;
        if(e->remaining_capacity < min_capacity) {
            printf("too small capacity %d %d\n", min_capacity, e->remaining_capacity);
            return false;
        }
        if(adj2[v].size()<=e->reverse_idx) {
            cout << "Bad index3!!!!!";
            return false;
        }
        e->remaining_capacity-=min_capacity;
        adj2[v][e->reverse_idx].remaining_capacity+=min_capacity;
        MinCostEdge er=adj2[v][e->reverse_idx];
        adj[u][min_cost_idxs[i]]=getAdj(*e, er, log_probability_cost_multiplier);
        adj[v][e->reverse_idx]=getAdj(er, *e, log_probability_cost_multiplier);

    }
    if(debug) cout << "adjusted cost after modification: "  << adj_total_cost(N, adj2) << "+" <<
        adj_total_mlog_prob(N, adj2) << "*" << log_probability_cost_multiplier << "=" <<
        adj_total_cost(N, adj2)+adj_total_mlog_prob(N, adj2)*log_probability_cost_multiplier
         << endl;
    // elapsed("decreased total cost rest", begin);

    return true;
}


int main(){
    // read simplified graph
    FILE *F=fopen("lightning.data", "r");
    if(!F) {return -1;}
    int N, M, s, t, value;
    int log_probability_cost_multiplier;
    fscanf(F, "%d%d%d%d%d%d", &N, &M, &s, &t, &value, &log_probability_cost_multiplier);  
    std::vector<OriginalEdge> lightning_data; // u, v, capacity, cost, flow=0
    char ss[1000];
    auto begin = now();
    for(int i=0; i<M; i++) {
        int u, v, capacity, cost, guaranteed_liquidity;
        fscanf(F, "%d%d%d%d%d", &u, &v, &capacity, &cost, &guaranteed_liquidity);
        if(!cost) {
            cost=1;
        }
        OriginalEdge oe {.u=u, .v=v, .capacity=capacity, .cost=cost, .flow=0, .guaranteed_liquidity=guaranteed_liquidity};
        lightning_data.push_back(oe);
    }
    elapsed("read", begin);

    // Find max path

    begin = now();
    MaxFlowGraph g(N);
    vector<int> edges_with_flow;
    for(int i=0; i<M; i++) {
        auto data=lightning_data[i];
        edges_with_flow.push_back(g.addEdge(data.u, data.v, data.capacity));
    }

    cout << "Maximum flow " << g.DinicMaxflow(s, t, value) << endl;
    elapsed("max flow", begin);
    begin = now();
    for(int i=0; i<M; i++) {
        int u=lightning_data[i].u;
        int flow = g.adj[u][edges_with_flow[i]].flow;
        lightning_data[i].flow=flow;
    }
    elapsed("edges_with_flow flow info", begin);
    cout << "Total cost before optimizations: " << total_cost(lightning_data)/1000000.0 << endl;
    int rounds=0;

    begin=now();
    std::vector<std::pair<int, int>> adj[N];  // v, cost
    std::vector<MinCostEdge> adj2[N];  // flow, capacity  // same for negative for now
    int numneg=0;
    std::vector<int> lightning_data_idx;
    for (int i = 0; i < lightning_data.size(); ++i)
    {
        auto data = lightning_data[i];
        int u=data.u, v=data.v;
        MinCostEdge e {data.v, data.capacity-data.flow, data.cost, (int) adj2[data.v].size(), data.guaranteed_liquidity};
        MinCostEdge er {data.u, data.flow, -data.cost, (int)  adj2[data.u].size(), data.guaranteed_liquidity};
        if(er.remaining_capacity > 0) {
            numneg++;
        }
        lightning_data_idx.push_back(adj2[data.u].size());
        adj2[u].push_back(e);
        adj2[v].push_back(er);
        adj[data.u].push_back(getAdj(e, er, log_probability_cost_multiplier));
        adj[data.v].push_back(getAdj(er, e, log_probability_cost_multiplier));
    }
    if(debug) {
        cout << "numneg: " << numneg <<endl;
        cout << "adj_total_cost: " << adj_total_cost(N, adj2)/value*100.0 << "%" << endl;
    }
    elapsed("setup early terminate", begin);
    long long cost_after_0=adj_total_cost(N, adj2)/1000000.0, cost_after_100=0, cost_after_200=0, cost_after_400=0;
    float  p_after_100=0, p_after_200=0, p_after_400=0;
    while(decrease_total_cost(N, adj, adj2, log_probability_cost_multiplier)) {
        auto distance=std::chrono::duration_cast<std::chrono::milliseconds>(now()-begin);
        if(cost_after_100==0 && distance.count()>100) {
            cost_after_100=adj_total_cost(N, adj2)/1000000.0;
            p_after_100=exp2(-adj_total_mlog_prob(N, adj2));
        }
        if(cost_after_200==0 && distance.count()>200) {
            cost_after_200=adj_total_cost(N, adj2)/1000000.0;
            p_after_200=exp2(-adj_total_mlog_prob(N, adj2));

        }
        if(cost_after_400==0 && distance.count()>400) {
            cost_after_400=adj_total_cost(N, adj2)/1000000.0;
            p_after_400=exp2(-adj_total_mlog_prob(N, adj2));
        }
        if(debug)
        cout << "total cost " << adj_total_cost(N, adj2)/1000000.0/value*100.0 << "%" << endl;
        rounds++;
        if(distance.count()>2000) {
            cout << "Breaking after 2s" << endl;
            break;
        }
    }
    cout << "Total cost after optimizations: " << adj_total_cost(N, adj2)/1000000.0/value*100.0 << "%, p=" << exp2(-adj_total_mlog_prob(N, adj2))*100 <<"%" << endl; // 0.1351%
    cout << "cost after 0 rounds: " << cost_after_0*1.0/value*100.0 << "%" << endl;  // 0.1404%
    cout << "cost after 100: " << cost_after_100*1.0/value*100.0 << "%, p=" << p_after_100*100 <<"%" << endl;  // 0.1404%
    cout << "cost after 200: " << cost_after_200*1.0/value*100.0 << "%, p=" << p_after_200*100 <<"%"<< endl;  // 0.1404%
    cout << "cost after 400: " << cost_after_400*1.0/value*100.0 << "%, p=" << p_after_400*100 <<"%" << endl;  // 0.1404%
    elapsed("total time", begin);  // 2500ms for 0.5 BTC
    cout << rounds << " rounds, satoshis=" << value << endl;
    printf("%d rounds\n", rounds);
    FILE *OUT=fopen("min_cost.out", "w");
    fprintf(OUT, "%ld\n", lightning_data.size());
    for(int i=0; i<lightning_data.size(); i++) {
        auto data =lightning_data[i];
        int u=data.u;
        auto e=adj2[u][lightning_data_idx[i]];
        auto er=adj2[e.v][e.reverse_idx];
        fprintf(OUT, "%d\n", er.remaining_capacity);
    }
    fclose(OUT);

}