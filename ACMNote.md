# ACM 模版总结

## 一.树

###1.1 树的基本知识:

* 基本术语
  1. 度：树种节点的**子节点**个数称为该节点的度，树中节点最大的度称为该树的度
  2. 树的路径长度：树的路径长度是根到所有节点的路径总和
  3. 完全二叉树：一棵高度为h的二叉树，含有$2^h - 1$个节点称为满二叉树

* 树的性质：
  1. 树中节点数等于节点度数 + 1 (特别的，对于非空二叉树，度数为0的节点个数等于度数为2的节点个数 + 1)
  2. 度为m的树第i层最多有 $ m^{i - 1} ​$ ( i >= 1)
  3. 高度为h的m叉树最多有$(m ^ h - 1) / (m - 1) $  (其实是个等比数列求和)


### 1.2 二叉树的存储

* 方案一：将树中所有节点存储于数组中，其子节点编号存放于节点结构体内部数组中

```
struct node
{
    int next[26]; // 字典树每个节点后续有sigma_size(这里是26个字母)个子节点，该next数组存储的是T中编号为i的子节点在T中的编号
    int val;
    void init()
    {
        val=0;
        memset(next,-1,sizeof(next)); // 将没有用过的子节点都标记为-1
    }
}T[1000000];

```
* 方案二：按照图的存储方案进行存储
* 方案三：该种方案仅用于二叉树，和方案一方法类似，但是其子节点利用完全二叉树编号体系(1 base，i节点的左子节点为2 * i, 又儿子为2 * i + 1 这个关系来进行维护）



### 1.3 判断是否是树
判断一个有向图是否是一棵树：
1. 有且仅有一个入度为0的点（根节点）
2. 除了跟节点之外的所有点的入度都必须为1 
3. 需要特判节点数为0的时候也是一棵树

```
#include <iostream>
#include <cstring>
#include <cstdio>
using namespace std;
const int maxn = 10000 + 10;

int in[maxn], n, cnt, vis[maxn];

void init(){
    memset(in, 0, sizeof(in));
    memset(vis, 0, sizeof(vis));
    n = 0;
}
/*
 * 判断一个有向图是否是一棵树，1.有且仅有一个入度为0的点（根节点）2.除了跟节点之外的所有点的入度都必须为1 3.需要特判节点数为0的时候也是一棵树
 */
int judge(){
    int root = -1;
    int root_cnt = 0;
    for(int i = 1; i <= n; i++){
        if(vis[i] && in[i] ==0){
            root_cnt++;
            root = i;
        }
    }
    if(root_cnt != 1) return 0;

    for(int i = 1; i <= n; i++){
        if(vis[i] && i != root){
            if(in[i] != 1) return 0;
        }
    }

    return 1;
}
int main(){
    freopen("input.txt", "r", stdin);
    int u, v;
    cnt = 0;
    init();
    while(scanf("%d%d", &u, &v)){
        if(u == -1 && v == -1) break;
        else if(u == 0 && v == 0){
            cnt++;
            if(judge() || n == 0) printf("Case %d is a tree.\n", cnt);
            else printf("Case %d is not a tree.\n", cnt);

            init();
        }
        else{
            n = max(n, u), n = max(n, v);
            vis[u] = 1, vis[v] = 1;
            in[v]++;
        }
    }
    return 0;
}
```

### 1.4 树的遍历(TODO)

#### 后续遍历

1. 例子：给定一个树的先序遍历和中序遍历，求解后序遍历
2. 解题思想：先序遍历中的首字符一定是树根，用这个根可以在中序遍历中找到根节点，之后根据中序遍历中左子树和右子树的长度可以确定先序遍历中的左子树和右子树，之后进行递归打印。
3. 如：
    * 先序遍历：ABDEFC
    * 中序遍历：DBEFAC
    * 首先确定该树的树根为A，之后可以在中序遍历中确定其左子树为DBEF，长度为4，之后就可以在先序遍历中确定左子树了。


```
#include <iostream>
#include <cstring>
#include <cstdio>
using namespace std;
char str_mid[30], str_pre[30];
void after(int ml, int mr, int pl, int pr){
	if(ml > mr || pl > pr){
		// printf("%c ", str_mid[ml]);
		return;
	}
	else{
		int root_pre = pl;
		int root_mid = -1;
		int len = -1;
		for(int i = ml; i <= mr; i++){
			if(str_mid[i] == str_pre[root_pre]){
				root_mid = i;
			}
		}
		len = root_mid - ml;
		after(ml, root_mid - 1, pl + 1, pl + len);
		after(root_mid + 1, mr, pl + len + 1, pr);
		printf("%c", str_mid[root_mid]);
	}
}
int main(){
	//freopen("input.txt", "r", stdin);
	scanf("%s", str_pre);
	scanf("%s", str_mid);
	int len = strlen(str_pre);
	after(0, len - 1, 0, len - 1);	
	return 0;
}
```

### 1.5 二叉树排序树
1. 二叉排序树或者是一棵空树，或者是具有下列性质的二叉树：
    * 若左子树不空，则左子树上所有结点的值均小于或等于它的根结点的值；
    * 若右子树不空，则右子树上所有结点的值均大于或等于它的根结点的值；
    * 左、右子树也分别为二叉排序树；
2. 经典例题：
    * 输入n个整数，构建二叉排序树，并进行前序列，中序列，后序遍历 ( [二叉排序树](https://www.nowcoder.com/practice/f74c7506538b44399f2849eba2f050b5?tpId=61&tqId=29557&tPage=3&ru=%2Fkaoyan%2Fretest%2F1002&qru=%2Fta%2Fpku-kaoyan%2Fquestion-ranking) )

### 1.6 哈弗曼树
1. 定义：二叉树T有t片树叶v1、v2。。vt, 其权值分别为w1、w2、。。wt,构造 $ W(T) = \sum_1^n{w_il(v_i)} $ 为二叉树T的权，其中 $l(v_i)$ 是节点$v_i $的层数（根节点为0层），权最小的二叉树被称为最优二叉树，求解最优二叉树的方法被称为Haffman算法。

3. 求解方法
    每次找出最小的两个节点，添加一个新的分支节点作为这两个节点的父节点，父节点的权值为两节点之和。重复这个步骤，直到只有一个节点为止。

4. 定理：
    * $$ W(T) $$等于所有分支节点的权之和。根据这个定理我们可以很轻松的解决求解$$ W(T) $$ 我们只需要每次累加分支节点的权值即可。
    * 一个分支节点的权值等于以其为根节点的子树的所有叶子节点的权值之和。

2. 经典例题
    * 给定Haffeman树的叶节点，构造Haffman树，并输出路径长度与节点权值之积的和(哈夫曼树)[https://www.nowcoder.com/practice/162753046d5f47c7aac01a5b2fcda155?tpId=67&tqId=29635&tPage=1&ru=/kaoyan/retest/1005&qru=/ta/bupt-kaoyan/question-ranking]，该题计算代码如下。

```
#include <iostream>
#include <cstring>
#include <cstdio>
#include <queue>
using namespace std;

const int maxn = 1000 + 10;

int n;

int main(){
    freopen("input.txt", "r", stdin);
    while(scanf("%d", &n) != EOF){
        int ans = 0;
        priority_queue <int, vector<int>, greater<int> > pq; // 采用优先队列
        for(int i = 1; i <= n; i++){
            int tmp;
            scanf("%d", &tmp);
            pq.push(tmp);
        }
        while(pq.size() > 1){
            int a = -1, b = -1;
            a = pq.top(), pq.pop(); // 每次从优先队列中取出最小的两个元素
            b = pq.top(), pq.pop();
            int c = a + b; // 添加新的分支节点
            ans += c; // 累加分支节点的权值
            pq.push(c);
        }
        printf("%d\n", ans);
    }
    return 0;
}

```

### 1.7 LCA最近公共祖先tarjan离线算法

* 算法步骤：

	1. 任选一个点为根节点，从根节点开始。
		
	2. 遍历该点u所有子节点v，并标记这些子节点v已被访问过。
		
	3. 若是v还有子节点，返回2，否则下一步。
		
	4. 合并v到u上(并查集操作merge)。
		
	5. 寻找与当前点u有询问关系的点v。
		
	6. 若是v已经被访问过了，则可以确认u和v的最近公共祖先为v被合并到的父亲节点a(并查集操作getf(v))。

* 算法思想：

  这里并查集的作用并不是要用来维护什么集合，而是用并查集的getf操作来获取lca。从任意一点为树根开始进行dfs会生成一个dfs树，那么考虑lca的两种典型情况：
  1. 两点在一条链上。对于第一种情况，当然只有dfs到深度比较深的那个节点v时候才开始进行lca查询，这时深度较浅的那个节点u因为还处于tarjan这个dfs中，还没有更改其父节点，所以getf(u) = u。
  2. 两点在不同的分叉上。首先我们设这两点u、v的lca(u, v) = a,当然只有dfs到第二个dfs节点时候才会进行lca查询，这时我们其实还处于节点a的tarjan的dfs中，这时候a的子树的所有getf()都为a，这时候查询到的getf(u)也一定为a，所以可以说明这个方法的正确行。

* 时间复杂度：时间复杂度为O(n + q)
* 伪代码：

```
	Tarjan(u)//marge和find为并查集合并函数和查找函数
{
    for each(u,v)    //访问所有u子节点v
    {
        Tarjan(v);        //继续往下遍历
        marge(u,v);    //合并v到u上
        标记v被访问过;
    }
    for each(u,e)    //访问所有和u有询问关系的e
    {
        如果e被访问过;
        u,e的最近公共祖先为find(e);
    }
}
```

* 应用题型：
	1. 第一种就是给一个树（要把边加双向），给q个两两点的lca，这类题（如Nearest Common Ancestors）画画图就知道如果根节点不一样，那么lac也会不一样，所以一定要从根节点开始进行tarjang操作
	2. 给一个树（可以为带权）求任意两点距离，因为有大量询问，如果用floyd需要O(n ^ 3)，Dijkstra需要O(n^2logn)，如果用dfs求出任意一点到达根节点距离，之后求出u, v的lca，那么这两点距离就是dis[u] + dis[v] - 2 * dis[lca(u, v)]
* 模板题：[Nearest Common Ancestors](http://poj.org/problem?id=1330)

```
#include <iostream>
#include <cstring>
#include <cstdio>
using namespace std;
const int maxn = 10000 + 10;
const int maxq = 100000 + 10;
struct edge
{
	int to;
	int next;
}Edge[2 * maxn];
struct node
{
	int to;
	int next;
	int lca;
	int order;
}Query[2 * maxn];
int head[maxn], n, pos; // 邻接表相关
int qhead[maxn], qpos; // 询问相关
int vis[maxn], f[maxn], root[maxn], ans[maxq];//用root来判断节点是否是根节点,根节点root为1，用ans来存储query的结果
void init(){
	memset(vis, 0, sizeof(vis));
	for(int i = 0; i < maxn; i++) head[i] = -1;
	for(int i = 0; i < maxn; i++) qhead[i] = -1;
	for(int i = 0; i < maxn; i++) f[i] = i;
	for(int i = 0; i < maxn; i++) root[i] = 1;
	qpos = 0;
	pos = 0;
}
void addEdge(int u, int v){ // 邻接表
	Edge[pos].to = v;
	Edge[pos].next = head[u];
	head[u] = pos;
	pos++;
}
void addQuery(int u, int v, int order){ // 利用和存图类似的方法存储询问
	Query[qpos].to = v;
	Query[qpos].next = qhead[u];
	Query[qpos].order = order;
	qhead[u] = qpos;
	qpos++;
}
int getf(int u){
	if(f[u] == u) return u;
	f[u] = getf(f[u]);
	return f[u];
}
void merge(int u, int v){ // 和一般并查集不同的是，这里的合并操作一定要将子树合并到父节点上，而不是相反
	int fu = getf(u);
	int fv = getf(v);
	f[fv] = fu;
}
void tarjan(int u){
	for(int k = head[u]; k != -1; k = Edge[k].next){
		int v = Edge[k].to;
		if(vis[v]) continue;
		vis[v] = 1;
		tarjan(v);
		merge(u, v); // 将当前节点的子树结合合并到该节点上
	}
	for(int k = qhead[u]; k != -1; k = Query[k].next){ // 查看有关该节点的所有询问，如果询问的另一个节点已经被访问过
		int v = Query[k].to;
		if(vis[v]){
			Query[k].lca = getf(v);
			ans[Query[k].order] = Query[k].lca;
		}
	}
}
int main(){
	//freopen("input.txt", "r", stdin);
	int T;
	scanf("%d", &T);
	for(int t = 1; t <= T; t++){
		scanf("%d", &n);
		init();
		int u, v;
		for(int i = 1; i <= n - 1; i++){
			scanf("%d%d", &u, &v);
			addEdge(u, v);
			addEdge(v, u);
			root[v] = 0;
		}
		scanf("%d%d", &u, &v);
		addQuery(u, v, 1); // 询问也要添加双向边
		addQuery(v, u, 1);
		for(int i = 1; i <= n; i++)
			if(root[i]){
				vis[i] = 1;
				tarjan(i);
			}
		printf("%d\n", ans[1]);
	}
	return 0;
}
```

### 1.8 树的直径 

* 树的直径：找出树相距最远的举例，连接这两点的树称为树的直径。
* 思路: 从任意一点开始进行dfs，找到距离最远的一个点u，之后从u点开始dfs，找到距离u点最远的的点v，那么从u到v的距离就是树的直径。
* 例题：[Cow Marathon](http://poj.org/problem?id=1985)[代码：](http://blog.csdn.net/u013555159/article/details/79490191)

```
#include <iostream>
#include <cstring>
#include <cstdio>
using namespace std;
const int maxn = 500000 + 10;
const int maxm = maxn + 10;
int head[maxn], pos, n, m, vis[maxn], dist[maxn];
struct edge
{
	int to;
	int next;
	int w;
}Edge[maxm];
void addEdge(int u, int v, int w){
	Edge[pos].to = v;
	Edge[pos].w = w;
	Edge[pos].next = head[u];
	head[u] = pos;
	pos++;
}
void init(){
	for(int i = 0; i < maxn; i++) head[i] = -1;
	pos = 0;
}
void dfs(int u, int d){
	vis[u] = 1;
	dist[u] = d;
	for(int k = head[u]; k != -1; k = Edge[k].next){
		int v = Edge[k].to;
		if(!vis[v]){
			dfs(v, Edge[k].w + d);
		}
	}
}
int getDiameter(){
	memset(vis, 0 , sizeof(vis));
	memset(dist, 0, sizeof(dist));
	dfs(1, 0); // 从任意一点开始进行dfs
	int u = -1, maximum = -1;
	for(int i = 1; i <= n; i++){
		if(dist[i] > maximum){
			maximum = dist[i]; // 找到距离最远的点u
			u = i;
		}
	}
	memset(vis, 0 , sizeof(vis));
	memset(dist, 0, sizeof(dist));
	dfs(u, 0); // 从u点开始进行dfs
	u = -1, maximum = -1;
	for(int i = 1; i <= n; i++){
		if(dist[i] > maximum){
			maximum = dist[i]; // 找到距离最远的点v，这个距离就是树的直径 
			u = i;
		}
	}
	return maximum;
}
int main(){
	//freopen("input.txt", "r", stdin);
	while(scanf("%d%d", &n, &m) != EOF){
		init();
		for(int i = 1; i <= m; i++){
			char temp;
			int u, v, w;
			scanf("%d%d%d", &u, &v, &w);
			getchar();
			getchar();
			//cout << u << v << w << endl;
			addEdge(u, v, w);
			addEdge(v, u, w);
		}
		int ans = getDiameter();
		printf("%d\n", ans);
	}
	return 0;
}
```


## 二.图论

### 2.1图的存储和遍历
#### 每个点边的数组存储法

```

#include <vector>
#include <queue>
using namespace std;
#define M 1009 //点的数量
#define INF 0x3f3f3f3f
struct edge
{
    int to,w;//保存边的信息，包括边的终点以及权值
};
vector<edge> g[M]; //利用一个vector保存，g[i]表示以i为起点的所有边的信息
int n,m,ee;

        for(int i = 0;i <= n;i++) //清空vector避免多kase相互影响
            g[i].clear();
        for(int i = 0;i < m;i++)
        {
            int a,b,c;
            scanf("%d %d %d",&a,&b,&c);
            edge e;
            e.to = b;e.w = c;
            g[a].push_back(e);
            //e.to = a;
            //g[b].push_back(e);  //题目中有说从a到b有这条
```


#### 每个点边的链表存储法
```C++
 const int N = 10000;//点的数量
 const int M = 10000;//边的数量
 int head[N],tot;
 struct node//结构体储存边信息
 {
    int to;//这条边的目标j
    int w;//边的权重
    int next;//指向上一条边
 }edge[M];//从0开始

 void init()
 {
    tot=0;
    memset(head,-1,sizeof(head));
 }

 void addedge(int i,int j)//添加边函数
 {
    edge[tot].to=j;
    edge[tot].next=head[i];
    head[i]=tot++;//和链表添加解点的方式一样
 }

```



### 2.2 DFS

1. 原理：参考算法p340
   用递归的方法遍历每个节点，在访问一个节点时：
    * 将它标记为已访问
    * 递归的访问它所有没有被标记过得邻居节点
2. 注解：单从函数实现角度看dfs可以认为dfs(k)是对状态k进行的操作，k就是对于状态的描述，从这个角度来考虑就不难实现dfs

```C++
void dfs(int k){ // dfs的参数其实是用来描述当前所在节点状态，可以以此为依据自行进行参数定义
	for(adj v){
		if(v is not visit){
			visit[v] = 1; // 将它标记为已访问
			dfs(v); // 递归的访问它所有没有被标记过得邻居节点
		}
	}
}
```


### 2.3 BFS

```C++
void bfs(int s){ 
	queue <int> q;
	q.push
}
```


### 2.3 并查集
```C++
const int maxn=11000;
int f[maxn];
void init()
{
    for(int i=1;i<maxn;i++) f[i]=i;
}
int getf(int a)
{
    if(f[a]==a) return a;
    f[a]=getf(f[a]);
    return f[a];
}

void merge(int a,int b)
{
    int fa=getf(a);
    int fb=getf(b);
    f[fa]=fb;
}

```

### 2.4 网络流

#### Dinic
添加边的时候要添加反向边

```
struct node {
    int to, next, f;
}Edge[maxm];
int head[maxn], level[maxn], N, F, D, pos;
void addEdge(int u, int v, int f){
    Edge[pos].to = v;
    Edge[pos].f = f;
    Edge[pos].next = head[u];
    head[u] = pos;
    pos++;
}
bool BFS(int s, int t){
    memset(level, 0, sizeof(level));
    level[s] = 1;
    queue <int> q;
    q.push(s);
    while(!q.empty()){
        int top = q.front();
        q.pop();
        if(top == t) return true;
        for(int k = head[top]; k != -1; k = Edge[k].next){
            if(Edge[k].f && !level[Edge[k].to]){
                level[Edge[k].to] = level[top] + 1;
                //cout<<top<<"_"<<k<<"_"<< Edge[k].to << "_" <<level[Edge[k].to]<< endl;
                q.push(Edge[k].to);
            }
        }
    }
    return false;
}
int DFS(int now, int maxf, int t){
    if(now == t) return maxf;
    int ret = 0;
    for(int k = head[now]; k != -1; k = Edge[k].next){
        if(Edge[k].f && level[Edge[k].to] == level[now] + 1){
            int f = DFS(Edge[k].to, min(Edge[k].f, maxf - ret), t);
            //cout << f <<"(";
            Edge[k].f -= f;
            Edge[k ^ 1].f += f;
            ret += f;
            if(ret == maxf) return ret;
        }
    }
    return ret;
}
int dinic(int s, int t){
    int ans = 0;
    while(BFS(s, t)){
        ans += DFS(s, inf, t);
    }
    return ans;
}
```

#### 最小费用流
* spaf:
设立一个先进先出的队列q用来保存待优化的结点，优化时每次取出队首结点u，并且用u点当前的最短路径估计值对离开u点所指向的结点v进行松弛操作，如果v点的最短路径估计值有所调整，且v点不在当前的队列中，就将v点放入队尾。这样不断从队列中取出结点来进行松弛操作，直至队列空为止

```
const int inf = 1 << 30;
const int MAXN = 200000 + 10;
const int MAXM = 1000000 + 10;
struct edge{
    int to;
    int next;
    int cost;
    int cap;
    int flow;
}Edge[MAXM];
int head[MAXN], pos, n;
void addEdge(int u, int v, int c, int cost){
    Edge[pos].to = v;
    Edge[pos].next = head[u];
    Edge[pos].cap = c;
    Edge[pos].flow = 0;
    Edge[pos].cost = cost;
    head[u] = pos;
    pos++;
    Edge[pos].to = u;
    Edge[pos].next = head[v];
    Edge[pos].cap = 0;
    Edge[pos].flow = 0;
    Edge[pos].cost = -1 * cost;
    head[v] = pos;
    pos++;
}
int dis[MAXN], vis[MAXN], pre[MAXN];

bool spfa(int s,int t)
{
    queue <int> q;
    for(int i = 0; i < MAXN; i++) {
        dis[i] = inf;
        vis[i] = false;
        pre[i] = -1;
    }
    dis[s] = 0;
    vis[s] = true;
    q.push(s);
    while(!q.empty()) {
        int u = q.front();
        q.pop();
        vis[u] = false;
        for(int i = head[u]; i != -1; i = Edge[i].next) {
            int v = Edge[i].to;
            if(Edge[i].cap > Edge[i].flow &&
                    dis[v] > dis[u] + Edge[i].cost ) {
                dis[v] = dis[u] + Edge[i].cost;
                pre[v] = i;
                if(!vis[v]) {
                    vis[v] = true;
                    q.push(v);
                }
            }
        }
    }
    if(pre[t] == -1)return false;
    else return true;
}
int minCostMaxflow(int s,int t,int &cost)
{
    int flow = 0;
    cost = 0;
    while(spfa(s,t)) {
        int Min = inf;
        for(int i = pre[t]; i != -1; i = pre[Edge[i^1].to]) {
            if(Min > Edge[i].cap - Edge[i].flow)
                Min = Edge[i].cap - Edge[i].flow;
        }
        for(int i = pre[t]; i != -1; i = pre[Edge[i^1].to]) {
            Edge[i].flow += Min;
            Edge[i^1].flow -= Min;
            cost += Edge[i].cost * Min;
        }
        flow += Min;
    }
    return flow;
}
void init(){
    for(int i = 0; i < MAXN; i++) head[i] = -1;
    pos = 0;
}
```


###2.5 拓扑排序
 拓扑排序
一.拓扑排序概念解释：
（1）针对的是有向无环图，无向图和有环图没有拓扑排序。
（2）针对一个有向无环图G，是将所有顶点排成一个线性序列，使得图中任意一对顶点u和v，若图G中存在<u,v>那么，在线性队列中u一定出现在v之前。这个线性队列称为满足拓扑次序的序列，简称拓扑序列。

二.算法原理：
 入度为零的边一定是拓扑排序的最前边的边。
三.算法流程：
（1）找到入度为0的点，输出
（2）把所有入度和从该点出发的边删去。
（3）循环上边两步，直到没有入度为0的点为止。
四.时间复杂度：O（m+n)


```
const int maxn=10000+10;
const int maxm=20000+10;

int head[maxn],ip,indegree[maxn];
int n,m,seq[maxn],cost[maxn];//如碰到HDU2647这种需要知道每个节点是属于拓扑排序的第几层的时候，cost[i]是记录i的层数的，从0层开始。

struct node
{
    int v,next;
} edge[maxm];

void init()
{
    memset(head,-1,sizeof(head));
    memset(cost,0,sizeof(cost));
    memset(indegree,0,sizeof(indegree));
    ip=0;
}

void addedge(int u,int v)
{
    edge[ip].v=v,edge[ip].next=head[u],head[u]=ip++;
    indegree[v]++;
}

int topo()//拓扑，可做模板
{
    queue<int>q;
    int indeg[maxn];
    for(int i=1; i<=n; i++)
    {
        indeg[i]=indegree[i];
        if(indeg[i]==0)
        {
            q.push(i);
            cost[i]=0;
        }
    }
    int k=0;//k用来记录删除的点的数量
    bool res=false;
    while(!q.empty())
    {
        if(q.size()!=1)res=true;
        int u=q.front();
        q.pop();
        seq[k++]=u;
        for(int i=head[u]; i!=-1; i=edge[i].next)  //
        {
            int v=edge[i].v;
            indeg[v]--;
            if(indeg[v]==0)
            {
                q.push(v);
                cost[v]=cost[u]+1;
            }
        }
    }
    if(k<n)return -1;///如果删除的点少于n，则存在有向环，总之不能进行拓扑排序
    if(res)return 0;///可以进行拓扑排序，并且只有唯一一种方式，seq数组即是排序完好的序列
    return 1;///可以进行拓扑排序，有多种情况，seq数组是其中一种序列
}
```

### 2.6 Tarjan计算强连通分量

* 强连通分量：在**有向图G**中，如果两个顶点vi,vj间（vi>vj）有一条从vi到vj的有向路径，同时还有一条从vj到vi的有向路径，则称两个顶点强连通(strongly connected)。如果有向图G的每两个顶点都强连通，称G是一个强连通图。有向图的强连通子图，称为强连通分量(strongly connected components)。

* 时间复杂度：O(E+V) 

* 算法思想：图参考算法竞赛入门经典白色p319，从任意一点开始dfs一定可以生成一个或多个dfs树，

* 算法步骤：从任意一点开始进行dfs，每次访问到一个节点需要进行两部操作：
	1. 记录dfs时间戳dfn 
	2. 将该点加入栈中（其实是一种保存子树的方法）3.计算该点能访问到的最早的父亲（dfn最小）节点记为low， 如果low == dfn 进行3，否则继续dfs
	3. 该点与其上的点（一个强连通分量）出栈，并将这些点标记为一个强连通分量（scc）
* 相关问题：
	1. 缩点：缩点后有向图一定会被转换成一个DAG（有向无环图），DAG有一些重要的性质，即如果想让该图成为一个强连通图，那么至少需要再添加max（a,b），a为入度为零的点的数量，b为出度为零的点的数量。另外要特别注意，如果只有一个点（或者一个强连通分量）存在，那么不需要再加边。
* 例题
	[1](http://blog.csdn.net/u013555159/article/details/52387069)
	[2](http://blog.csdn.net/u013555159/article/details/52402309)
	[模板题](http://blog.csdn.net/u013555159/article/details/79466577)
2. 
怎么实现缩点的效果呢？（怎么将一个强连通分量当作一个点来处理）
//缩点完后调用为
```
    find(1,n);
        scc();
        cnt为总共的点数
```

```
const int N = 1E4 + 10;
vector<int>g[N];
int low[N]; //low存贮当前节点回溯所能找到栈中最早的节点。
int dfn[N]; //DFS时间戳
int dfs_clock;
int sccon[N],cnt;
bool instack[N];
int in[N],out[N];
stack<int>s;
int n,m;
//tarjan主要维护了两个数组，一个是low，一个是dfn，dfn记录了dfs时间戳，low记录了从当前节点所能回到的栈中时间戳最小的点，如果一个节点点dfn和low相等那么该点一定是一个强连通分量树点根，强连通分量的数量cnt+   +。

// 这里栈其实维护了一个没有强连通分量子树的dfs树，当dfs树找到一个强连通分量后，就会将该强连通分量pop出去，以维护dfs树中没有强连通分量子树
void tarjan(int u,int fa) {
    int v;
    low[u]=dfn[u]=++dfs_clock; //先将low和dfn都置为时间戳
    s.push(u);
    instack[u]=true;
    for(int i=0; i<g[u].size(); i++) { // 向四周进行dfs，实际是计算点low值
        v=g[u][i];
        if(!dfn[v]) { // 如果该点没有被访问过，进行dfs，并更新low
            tarjan(v,u);
            low[u]=min(low[u],low[v]);
        } else if(instack[v]) // 如果该点被访问过，并且在栈中，更新low(就是访问到了祖先节点了)
            low[u]=min(low[u],dfn[v]);
    }
    if(low[u]==dfn[u]) { 
        cnt++;
        while(1) {
            v=s.top()
            s.pop();
            instack[v]=false;
            sccon[v]=cnt;
            if(v==u)
                break;
        }
    }
}
// 使用方法：首先调用find
// find会从所有点进行dfs，生成一棵dfs树
void find(int l,int r) {
    memset(low,0,sizeof(low));
    memset(dfn,0,sizeof(dfn));
    memset(sccon,0,sizeof(sccon));
    memset(instack,false,sizeof(instack));
    dfs_clock=cnt=0;
    for(int i=l; i<=r; i++)
        if(!dfn[i])
            tarjan(i,-1);
}
// 使用效果：缩点，就是将强连通分量和强连通分量之外的连接关系都连接在强连通分量的根节点上
// 使用方法：调用完find之后使用
void scc() {
    int i,j;
    for(i=1; i<=cnt; i++)//设置每个强连通分量的出入度都为0
        in[i]=out[i]=0;
    for(i=1; i<=n; i++) {//对于每个点
        for(j=0; j<g[i].size(); j++) { //遍历便利每条边
            int u=sccon[i];//sccon可以将每个点映射到其所在强连通分量的根节点上
            int v=sccon[g[i][j]];
            if(u!=v) {
                out[u]++;
                in[v]++;
            }
        }
    }
}

```

```
#include <iostream>
#include <cstring>
#include <stack>
#include <cmath>
using namespace std;
const int maxn = 10000 + 10;
const int maxm = 100000 + 10;
int head[maxn], pos, n, m; // 邻接表
int low[maxn], dfn[maxn], dfs_clock, sccon[maxn], cnt, instack[maxn];
int in[maxn], out[maxn];
stack <int> s;
struct edge
{
	int to;
	int next;
}Edge[maxm];
void addEdge(int u, int v){
	Edge[pos].to = v;
	Edge[pos].next = head[u];
	head[u] = pos;
	pos++;
}
void init(){

	// 初始化邻接表
	pos = 0;
	for(int i = 0; i < maxn; i++) head[i] = -1;

	// 初始化tarjin
	while(!s.empty()) s.pop();
	memset(dfn, 0, sizeof(dfn));
	memset(low, 0, sizeof(low));
	memset(instack, 0, sizeof(instack));
	memset(sccon, 0, sizeof(sccon));
	dfs_clock = 0;
	cnt = 0;

}
void tarjin(int u, int fa){
	dfn[u] = low[u] = ++dfs_clock;
	s.push(u);
	instack[u] = 1;
	for(int k = head[u]; k != -1; k = Edge[k].next){ // 递归搜索并计算该点的low值
		int v = Edge[k].to;
		if(!dfn[v]){
			tarjin(v, u);
			low[u] = min(low[u], low[v]);
		}else if(instack[v]){
			low[u] = min(low[u], dfn[v]);
		}
	}
	if(dfn[u] == low[u]){ // 当且仅当一个点的dfn == low时该点是强连通分量的dfs树树根
		cnt++;
		while(1){
			int v = s.top();
			s.pop();
			instack[v] = 0;
			sccon[v] = cnt;
			if(v == u) break;
		}
	}
}
void find(int l, int r){
	for(int i = l; i <= r; i++){ // 从一个没有被dfs过的点开始进行dfs，生成dfs树
		if(!dfn[i])
			tarjin(i, -1);
	}
}
void scc(){
	for(int i = 1; i <= cnt; i++){
		in[i] = 0;
		out[i] = 0;
	}
	for(int i = 1; i <= n; i++){
		for(int k = head[i]; k != -1; k = Edge[k].next){
			int v = sccon[Edge[k].to];
			int u = sccon[i];
			if(u != v){
				out[u]++;
				in[v]++;
			}
		}
	}
}
```

### 2.7 最短路径问题

#### spfa

```C++
#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <queue>
using namespace std;
#define M 1009
#define INF 0x3f3f3f3f
struct edge
{
    int to,w;//保存边的信息，包括边的终点以及权值
};
int dis[M];  //最短距离的估计值（当前该点的最短距离）
bool inq[M]; //标记该点是否在队列之中
vector<edge> g[M]; //利用一个vector保存，g[i]表示以i为起点的所有边的信息
int n,m,ee;
void spfa(int u)
{
    for(int i = 0;i <= n;i++) //初始化
    {
        dis[i] = INF; //将估计值都初始化为INF
        inq[i] = false; //初始化为不在队列中
    }
    dis[u] = 0; //起点的估计值直接就是0
    inq[u] = true; //加入队列并进行标记
    queue<int> q;
    q.push(u);
    while(!q.empty())
    {
        u = q.front();
        inq[u] = false;
        q.pop();
        for(int i = 0;i < g[u].size();i++)
        {
            int v = g[u][i].to; //找出这条边对应的终点
            int w = g[u][i].w;  //这条边对应的权值
            if(dis[v] > dis[u]+w) //如果终点的最短距离比起点的最短距离加上这条边的权值那么就更新
            {
                dis[v] = dis[u]+w;
                if(!inq[v]) //如果v点的最短距离有所更新并且不在队列中，就将其加入队列。
                {           //否则就不需要重复加入队列增加不必要的操作。
                    inq[v] = true;  //加入队列并标记
                    q.push(v);
                }
            }
        }
    }
}
int main()
{
   while(scanf("%d %d %d",&n,&m,&ee)==3)
    {
        for(int i = 0;i <= n;i++) //清空vector避免多kase相互影响
            g[i].clear();
        for(int i = 0;i < m;i++)
        {
            int a,b,c;
            scanf("%d %d %d",&a,&b,&c);
            edge e;
            e.to = b;e.w = c;
            g[a].push_back(e);
            //e.to = a;
            //g[b].push_back(e);  //题目中有说从a到b有这条路，并不是双向的。
        }
        int s;
        scanf("%d",&s);
        for(int i = 0;i < s;i++)
        {
            int a;
            scanf("%d",&a);
            edge e;
            e.to = a; e.w = 0;
            g[0].push_back(e); //将家里到起点的权值设为0
            //e.to = 0;
            //g[a].push_back(e);  //从起点到家里是否有路其实无关紧要，因为思考一下有路的话 起点到起点也是0，所以根本不影响结果
        }
        spfa(0);
        //printf("debug-----%d\n",dis[ee]);
        if(dis[ee]==INF)
            printf("-1\n");
        else
            printf("%d\n",dis[ee]);
    }
}

```

#### Dijkstra算法 + 路径打印
* 适用性
  Dijkstra算法适用于 边权重非负的、加权有向图的单起点路径问题——算法p421（环？无向？是否适用？）
* 算法思想

  将图G中的所有顶点V分成两个集合Va、Vb，如果源点S到u的最短路已经确定，则u属于Va，否则属于Vb。开始Va只包含源点S。

* 算法步骤

  有两个集合需要维护，一个是源点到其他点的距离dist，另一个是集合Va，Vb。

  1. 将源点加入Va，
  2. 从Vb中找一点到S距离最短的点u从Vb中去除并加入Va中。
  3. 用u更新Vb中所有点的距离（u其实只能更新与u相临的点）。

* 应用
  * 路径打印

    利用pre[i]记录每个点的前驱节点实现路径打印

  * 统计最短路径数量

  * 最短路且有特殊要求

      1. 边特殊要求
         将图中的点分成两类（1类和2类），寻找1号点到n号点的最短路径，并且最多只能经过一个穿越边（从1类到2类或从2类到1类的）。解题思想是枚举穿越边。首先用dijskras处理1号点到同类点的最近距离(处理过程中不能通过不同类的节点)，同理处理n号点到同类的距离，之后（如果没有直接算出来1到n的距离的话）枚举穿越边(u, v)，距离就是w(u, v) + dist[1][u] + dist[n][v]    例题:[I wanna go home](https://www.nowcoder.com/practice/0160bab3ce5d4ae0bb99dc605601e971?tpId=61&tqId=29500&tPage=1&ru=/kaoyan/retest/1002&qru=/ta/pku-kaoyan/question-ranking) 

  * 在非DAG上进行dp求解

* 典型例题

 1. [紧急救援](https://www.patest.cn/contests/gplt/L2-001) [题解](http://blog.csdn.net/u013555159/article/details/79437751)
 2. 最短路径且要求最多智能经过一条特殊边[I wanna go home](https://www.nowcoder.com/practice/0160bab3ce5d4ae0bb99dc605601e971?tpId=61&tqId=29500&tPage=1&ru=/kaoyan/retest/1002&qru=/ta/pku-kaoyan/question-ranking) [题解](http://mp.csdn.net/postedit/79612235)

```C++
// 没有采用优先队列进行优化，为n^2，有路径打印
const int maxn = 500 + 10;
const int maxm = 1000000 + 10;
const int inf = 1 << 30;
struct Edge
{
	int v;
	int w;
	int next;
}edge[2 * maxm];
int head[maxn], pos;
int dist[maxn], pre[maxn], pathNum[maxn]; // 每个点需要维护的信息 dist为距离源点的距离，pre是该点的前驱节点，pathNum是从源点s到达该点的路径数量
int n, m;
void addEdge(int u, int v, int w){
	edge[pos].v = v;
	edge[pos].w = w;
	edge[pos].next = head[u];
	head[u] = pos;
	pos++;
}
void init(){
	for(int i = 0; i < maxn; i++) head[i] = -1;
	for(int i = 0; i < maxn; i++) pre[i] = -1;
	for(int i = 0; i < maxn; i++) pathNum[i] = 0;
	for(int i = 0; i < maxn; i++) getPeople[i] = 0;
	pos = 0;
}
void Dijkstra(int s, int t){
	for(int i = 0; i < n; i++) dist[i] = inf; // 初始化s到其他点的举例
	// 初始化s
	dist[s] = 0;
	pathNum[s] = 1; 
	pre[s] = -1;

	int V[maxn]; // V为0的是Vb集合元素，V为1是Va集合元素
	memset(V, 0, sizeof(V));
	for(int i = 0; i < n; i++){ // 每次选取Vb中最近的点加入Va（n个点）
		int u = -1, minimum = inf;
		for(int j = 0; j < n; j++){ // 选取Vb中dist最近的点
			if(!V[j] && dist[j] < minimum){
				minimum = dist[j];
				u = j;
			}
		}
		V[u] = 1; // 加入Va
		//用u更新Vb中所有和u相邻的点的距离和其他信息
		for(int k = head[u]; k != -1; k = edge[k].next){
			int v = edge[k].v;
			if(!V[v]){
				if(dist[v] > edge[k].w + dist[u]){
					dist[v] = edge[k].w + dist[u];
					pre[v] = u;
					pathNum[v] = pathNum[u];
				}
				else if(dist[v] == edge[k].w + dist[u]){
					pathNum[v] += pathNum[u];
				}
			}
		}
	}
}

void printPath(int s, int t){
	int road[maxn];
	int p = t;
	int cnt = 0;
	while(p != -1){
		road[cnt] = p;
		p = pre[p];
		cnt++;
	}
	for(int i = cnt - 1; i >= 0; i--){
		printf("%d", road[i]);
		if(i != 0) printf(" ");
	}
	printf("\n");
}
```


```C++
// 优先队列优化nlongn，无路径打印简单版
void dijkstra(int s) {
    priority_queue<pa,vector<pa>,greater<pa> >q;
    int i, now;

    for (i = 1; i <= n; i++)
        dis[i] = INF;
    dis[s] = 0;
    q.push(make_pair(0,s));
    while (!q.empty()) {
        now = q.top().second;
        q.pop();
        for (i = head[now]; i != -1; i = edge[i].next){
            if (dis[now] + edge[i].v < dis[edge[i].to]) {
                dis[edge[i].to] = dis[now] + edge[i].v;
                q.push(make_pair(dis[edge[i].to], edge[i].to));
            }   
        }

    }
}
```



```C++
/*
补充：DJS求解DP，给定一张带权图，可以使得图中不超过k边权值变成0，求解从1点到n点最短路径值
解：dp[i][k] = min(dp[j][k] + dis[i][j], dp[j][k - 1]), dp[i][k]定义为从1点到i点使得k条边权值变为0的最短路径值
注：这个dp方程之所以用dijskra算法求解，是因为其思想是和dijskra算法是一致的，每次通过优先队列取出的状态一定是该状态的最小值，利用这个状态去更新其他的状态
*/
void Djs(){
    priority_queue <ele> Q;
    for(int j = 0; j <= k; j++){
    	dp[1][j] = 0;
    	Q.push((ele){1, j, dp[1][j]});
    }
    while(!Q.empty()){
        int u = Q.top().u;
        int s = Q.top().s;
        Q.pop();
        for(int i = head[u];i != -1;i = edge[i].next){
           	int  v = edge[i].to;
           	ll dis = edge[i].w;
            if(dp[u][s] + dis < dp[v][s]){
            	dp[v][s] = dp[u][s] + dis;
            	Q.push((ele){v, s, dp[v][s]});
            }

            if(dp[u][s] < dp[v][s + 1] && s + 1 <= k){
            	dp[v][s + 1] = dp[u][s];
            	Q.push((ele){v, s + 1, dp[v][s + 1]});
            }

        }
	}
}
```





#### floyd算法

```C++
#include<cstdio>
#include<iostream>
using namespace std;
#define MAX 500
#define INFE 1<<20
 
int N; 
int map[MAX][MAX],b[MAX],path[MAX][MAX];  //path[i][j]记录路径
 
void init()
{
       int i,j;
       for(i=1;i<=N;i++)
              for(j=1;j<=N;j++)
              {
                     map[i][j]=INFE;
                     path[i][j]=j;
              }
}
 
void floyd()
{
       int i,j,k;
       for(k=1;k<=N;k++)
              for(i=1;i<=N;i++)
                     for(j=1;j<=N;j++)
                            if(map[i][j]>map[i][k]+map[k][j])
                            {
                                   map[i][j]=map[i][k]+map[k][j];
                                   path[i][j]=path[i][k];
                            }
}
 
 
int main()
{
       int m,u,v,len;
       while(scanf("%d%d",&N,&m)!=EOF) //输入城市数量 和 道路数量
       {
              init();//初始化
              while(m--)
              {
                     scanf("%d%d%d",&u,&v,&len);
                     map[u][v]=map[v][u]=len;
              }
              floyd();//进行每对节点的求最小路径
              
              while(scanf("%d%d",&u,&v))
              {//输入起点和终点
                     int tmp=u;
                     printf("%d",u);
                     while(tmp!=v)
                     {//打印路径
                            printf("-->%d",path[tmp][v]);
                            tmp=path[tmp][v];
                     }
                     //输出最小花费
                     cout<<endl;
                     cout<<"cost: "<<map[u][v]<<endl;
              }
       }
       return 0;
}
```

## 三.数论
### 3.1欧拉函数计算
1. 讲解:欧拉函数 $\varphi(n)$ 是不超过n且与n互质的正整数个数
2. 函数性质:
### 3.2质因数分解

```
struct node
{
	int factor;
	int cishu;
};

void factor(int n, node A[], int& len){
	int temp, i, now;
	temp = (int)((double)sqrt(n) + 1);
	len = 0;
	now = n;
	for(int i = 2; i <=temp;i++){
		if(now % i == 0){
			A[len].factor  = i;
			A[len].cishu = 0;
			while(now % i == 0){
				A[len].cishu++;
				now /= i;
			}
			len++;
		}
	}	
	if(now != 1){
		A[len].factor  = now;
		A[len].cishu = 1;
		len++;
	}
}
```

### 3.3最大公约数、最小公倍数
```
int gcd(int a,int b)//最大公约数
{
    if(b==0) return a;
    else return gcd(b,a%b);
}

int lcm(int a,int b)//最小公倍数
{
    return a/gcd(a,b)*b;
}
```

### 3.4拓展欧几里得算法

[算法详解](http://blog.csdn.net/u013555159/article/details/52297220)
* 接口
 ax+by=gcd(a,b)
 input:exgcd(a,b,x,y)
 output:(x0,y0)(一组可行解)
* 常见问题：
（1）合理解计算方法：x=x0+kb' y=y0+ka' ( b'=b/gcd(a,b) a'=a/gcd(a,b) )
（2）推广：

```
ax+by=c
a*[gcd(a,b)/c*x]+b*[gcd(a,b)/c*y]=gcd(a,b)
exgcd(a,b,tx0,ty0)
x=c/gcd(a,b)tx0+c/gcd(a,b)*k(b/gcd(a,b))
```

可以利用exgcd求出来tx0，然后换算成x后，直接进行+kb‘的计算


```
ll exgcd(ll a,ll b,ll &x,ll &y)
{
    ll r,t;
    if(!b){ x=1; y=0; return a;}
    r=exgcd(b,a%b,x,y);
    t=x;
    x=y;
    y=t-a/b*y;
    return r;
}
```
### 3.5模运算
1.加法:(a + b) % mod = (a % mod  + b % mod) % mod 
2.减法:(a - b) % mod = ((a % mod + b % mod) % mod + mod) % mod
3.乘法:(a * b) % mod = (a % mod * b % mod) % mod
4.除法:(a / b) % mod = (a % mod * inv(b)) % mod
### 3.6逆元
1. 讲解：$ (a / b) % p != (a % p / b % p) $, 所以我们要用到逆元来求解，ax == 1（mod p) x就是a的逆元， a / b == a * 1/b(mod p) 当且仅当b、p互质，才能用这种方法求解。可以采用拓展欧几里得来求解
(a / b) % p = (a * inv(b) ) % p = (a % p * inv(b) % p) % p

```
ll extgcd(ll a, ll b, ll &x, ll &y){
 ll d = a;
	if(b != 0){
		d = extgcd(b, a % b, y, x);
		y -= (a / b) * x;
	} else {
		x = 1;
		y = 0;
	}
	return d;
}
ll inv (ll a, ll m){ // m 是 mod
	ll x, y;
	extgcd(a, m, x, y);
	return (m + x % m) % m;
}
```


### 3.7埃氏筛法

1. 原理：
 如果判断出一个数是素数，那么以这个数为因子的数就一定不是素数。并且如果n是一个合数那么n一定有一个不超过sqrt（n）的素数因子。
2. 接口

```
 int N; 数据范围
 bool isprime[N];
 int prime[N];
 int nprime[N];
 void doprime();
const int N = 10000;
bool isprime[N];
int prime[N];
int nprime;
void doprime()
{
    memset(isprime,1,sizeof(isprime));
    nprime=0;
    for(int i=2;i<N;i++)
    {
        if(isprime[i])
        {
            prime[++nprime]=i;
            for(int j=i*i;j<N;j+=i) isprime[j]=false;
        }
    }
}
```

### 3.8快速幂取模

```
int quickpow_mod(int a, int n, int mod){
    if(n == 0) return 1;
    int x = quickpow_mod(a, n / 2, mod);
    long long ans = (long long)x * x % mod;
    if(n % 2 == 1) ans = ans * a % m;
    return (int)ans;
}
```

#### 3.7强连通分量Kosaraju

1. 使用方法：
    init（）初始化
    addedge（）加边
    Kosaraju(int n)计算
    belg[N], num[N]是计算结果
2. 时间复杂度：

3. 问题：
 （1）适用于什么图？有向图？无向图？如果是都可以的话无向图需要正反边分别使用addedge添加么？

```
const int N =10000+10, M=100000+10;
struct node
{
    int to, next;
}edge[M],edge2[M]; //edge是逆图,edge2是原图
int  dfn[N], head[N], head2[N],  belg[N], num[N];
//dfn时间戳
//belg记录每个点属于哪个连通分量。num记录每个强连通分量点的个数,其下表和belg中对应。
bool  vis[N];
int cnt,cnt1,scc,tot,tot1;
void dfs1(int u)
{
    vis[u]=1;
    for(int k=head2[u];k!=-1;k=edge2[k].next)
        if(!vis[edge2[k].to]) dfs1(edge2[k].to);
    dfn[++cnt1]=u;
}
void dfs2(int u)
{
    vis[u]=1;
    cnt++;
    belg[u]=scc;
    for(int k=head[u];k!=-1;k=edge[k].next)
        if(!vis[edge[k].to]) dfs2(edge[k].to);
}
void  Kosaraju(int n)
{
    memset(dfn,0,sizeof(dfn));
    memset(vis,0,sizeof(vis));
    cnt1=scc=0;
    for(int i=1;i<=n;i++)
        if(!vis[i]) dfs1(i);
    memset(vis,0,sizeof(vis));
    for(int i=cnt1;i>0;i--)
        if(!vis[dfn[i]])
        {
            cnt=0;
            ++scc;
            dfs2(dfn[i]);
            num[scc] = cnt;
        }
}
void init()
{
    tot=tot1=0;
    memset(head,-1,sizeof(head));
    memset(head2,-1,sizeof(head2));
    memset(num,0,sizeof(num));
}
void addedge(int i,int j)
{
    edge2[tot1].to=j; edge2[tot1].next=head2[i];head2[i]=tot1++;
    edge[tot].to=i; edge[tot].next=head[j];head[j]=tot++;
}

```

## 四.序列问题

### 最大子段和

### 逆序对

### 最长公共子序列

### 最长上升子序列

### 最长公共上升子序列
### 最长回文子序列
* 区分子串和子序列：子串一定是连续的，子序列一定是连续的


```
int longestPalindromeSubSequence2(string str){
    int n=str.length();
    vector<vector<int> > dp(n,vector<int>(n));

    for(int i=n-1;i>=0;i--){
        dp[i][i]=1;
        for(int j=i+1;j<n;j++){
            if(str[i]==str[j])
                dp[i][j]=dp[i+1][j-1]+2;
            else
                dp[i][j]=max(dp[i+1][j],dp[i][j-1]);
        }
    }
    return dp[0][n-1];
}
```
### 最长回文子串
1. 问题：给定一个字符串，找到该字符串中最长的回文子串(串是连续的子序列)
2. Manacher算法O(n)：
    * 该算法的基本思路是枚举回文串中心，并计算以该位置为中心的回文串长度（偶串经过添加字符处理即可转换为奇串情况）。
    * Manacher算法主要是在上面的方法中进行一定的改进。当我们枚举回文中心，计算以该位置为中心的回文长度时，可以通过已经计算过的回文串，来确定该位置为中心回文串的下限，之后从下限开始进行半径拓展即可。
    * 结合代码对该算法进行解释，mx是已经匹配完毕的结尾最远的回文串”到达了Ma[]数组的第Mx位。Id是已经匹配完毕的结尾最远的回文串”中心为Ma[]数组的第ID位。
    * 解释最核心的 mx>i?min(Mp[2 * id -i], mx - i):1; 如果当前计算的以i为中心的回文串，i的已经被之前计算过的回文串覆盖了(mx > i)，那么我们就可以利用之前的回文串性质，找到i位置的字符在之前的回文串中对应的回文的位置(我们叫y)。根据回文的性质，我们可以利用以y为中心的回文串长度，确定以i为中心的回文串长度的下限。Mp[2 * id -i] 代表以y为中的回文串长度， 但是还要考虑如果这个长度超过了其这个回文串的长度，那么，我们要把下限限制在这个回文串当中。
    * (知乎讲Manacher)[https://www.zhihu.com/question/37289584], (SegmentDefualt)[https://segmentfault.com/a/1190000003914228]
3. 经典题目： (hiho一下第一周)[http://hihocoder.com/contest/hiho1/problem/1]

```
/*
hiho 第一周：最长回文子串
ps:
	子串一定是连续的、序列不一定连续
*/


#include <iostream>
#include <cstring>
#include <cstdio>
using namespace std;

const int maxn = 1e6 + 10;

int origin_len, Ma_len; // 分别为原串长度，处理后长度
int Mp[3 * maxn];       // 以i为中心的回文子串长度半径
char origin[maxn], Ma[3 * maxn]; // 原串和处理后串

void Manacher(){
	int mx = 0, id = 0;
	for(int i = 0; i < Ma_len; i++){
		Mp[i] = mx>i?min(Mp[2 * id -i], mx - i):1;
		while(Ma[i + Mp[i]] == Ma[i - Mp[i]]) Mp[i]++; // 在下限基础上进行拓展
		if(Mp[i] + i > mx){ // 更新最远距离
			mx = i + Mp[i];
			id = i;
		}
	}
}

int main(){
	//freopen("input.txt", "r", stdin);
	int T;
	scanf("%d", &T);
	while(T--){
		scanf("%s", origin);

		origin_len = strlen(origin);
		Ma[0] = '$';
		Ma[1] = '#';
		for(int i = 0; i < origin_len; i++){
			Ma[2 * i + 2] = origin[i];
			Ma[2 * i + 3] = '#';
		}
		Ma_len = 2 * origin_len + 2;
		Ma[Ma_len] = '*';

		Manacher();

		int ans = 0;
		for(int i = 0; i < Ma_len; i++){
			ans = max(ans, Mp[i]);
		}

		printf("%d\n", ans - 1);

	}

}
```
## 五.字符串问题

### 1.9 Trie树
1. 定义：Trie树也叫字典树，可以将很多个字符串，以树的结构进行存储，树中每个节点是一个字符。支持插入和查询两种操作。从0号节点出发，到叶子节点，经过的所有字符就是存储在字典树中的字符串。![y](/Users/zhangjiatao/ACM/ACMNote/1.jpg)    [讲解](http://hihocoder.com/contest/hiho2/problem/1)/Users/zhangjiatao/ACM/ACMNote/1.jpg

2. 代码讲解：
    * 该模板的代码以树中的节点作为一个结构体即node, 并将其可能的子节点的编号存储在next[sigma_size]数组当中，val用于存储节点关键信息(信息内容依据题目而定)
    * 整个Trie树是由node结构体节点连接构成的，支持查询和插入两种操作
3. 典型题目：[题1](http://hihocoder.com/contest/hiho2/problem/1)

```
/*
hiho coder第二周
主要思想就是构建一个字典树，并在字典树上进行统计运算
*/
#include<cstdio>
#include<cstring>
#include<iostream>
using namespace std;
struct node
{
    int next[26]; // 字典树每个节点后续有sigma_size(这里是26个字母)个子节点，该next数组存储的是T中编号为i的子节点在T中的编号
    int val;
    void init()
    {
        val=0;
        memset(next,-1,sizeof(next)); // 将没有用过的子节点都标记为-1
    }
}T[1000000];

int size;

void insert(char *s){ //插入操作

    int p = 0, len =  strlen(s); // p是当前遍历所处位置

    for(int i = 0; i < len; i++){
        int c = s[i] - 'a'; // 找到当前字符所对应子节点编号
        if(T[p].next[c] == -1){ //如果该节点没有访问过，那么进行初始化
            T[size].init();
            T[p].next[c] = size;
            size++;
        }
        p = T[p].next[c];
        T[p].val++;
    }
}

int query(char *s){
    int p = 0, len = strlen(s);
    for(int i = 0; i < len; i++){
        int c = s[i] - 'a';
        if(T[p].next[c] == -1){
            return 0;
        }
        p = T[p].next[c];
    }
    return T[p].val;
}

int main()
{
    //freopen("input.txt", "r", stdin);
    int n,m;
    char str[20];
    while(scanf("%d",&n) != EOF){

        size=1;
        T[0].init();

        for(int i = 0; i < n; i++){
            scanf("%s",str);
            insert(str);
        }
        scanf("%d",&m);
        //cout << n <<" "<< m << endl;
        for(int i = 0; i < m; i++){
            scanf("%s",str);
            int ans = query(str);
            cout << ans << endl;
        }
    }
}
```
### 5.1KMP算法
1. 概述：将被匹配的串称作原串T，匹配称为模式串P
2. 算法思想：KMP主要思想是将模式串构造成一个自动机,自动机结构在大白书训练手册P211。getFail函数就是在构造自动机的失配边(如果在自动机的状态i未匹配成功，将如何进行转移)。

```
#include <iostream>
#include <cstdio>
#include <cstring>
using namespace std;

const int max_len = 1e6 + 10;

int f[max_len];
char str[max_len], p[max_len];

//自动机失配边构造
void getFail(char *P){
	int len_P = strlen(P);
	f[0] = 0;
	f[1] = 0;
	for(int i = 1; i < len_P; i++){
		int pos = f[i];
		while(pos && P[pos] != P[i]) pos = f[pos];
		f[i + 1] = P[i] == P[pos] ? pos + 1 : 0;
	}
}

int KMP(char *T, char *P){

	getFail(P); // 构造适配边
	int cnt = 0;
	int len_T = strlen(T), len_P = strlen(P);
	int pos = 0; // 自动机状态位置
	for(int i = 0; i < len_T; i++){
		while(pos && P[pos] != T[i]) pos = f[pos]; // 如果不匹配就一直沿着失配边走，直到匹配, 停止条件为1.匹配成功2.已经到达自动机起点
		if(P[pos] == T[i]) pos++; //判断之前是否是匹配成功而停止
		if(pos == len_P){ // 找到匹配位置
			cnt++; 
		}
	}
	return cnt;
}


int main(){
	freopen("input.txt", "r", stdin);
	int T;
	scanf("%d", &T);
	while(T--){
		scanf("%s%s", p, str);
		int ans = KMP(str, p);
		printf("%d\n", ans);
	}
	return 0;
}
```
### 5.2AC自动机
1. 问题： 给定一个模板串和*多个*模式串，在模板串中找到模式串
2. 算法讲解：缺省后补！！！！
3. 题目：
    * 模板题:[Trie图](http://hihocoder.com/problemset/problem/1036?sid=1313540)

```
//不要求输出匹配到的模式串编号版本
#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <queue>
#include <vector>

using namespace std;

const int maxn = 1000010; // Trie图最大节点个数

struct AC
{	
	int last[maxn];
	int f[maxn];
	int ch[maxn][26];
	int val[maxn];
	int sz;
	void init()
	{
		sz = 1; memset(ch[0],0,sizeof(ch[0]));
	}
	int idx(char c) { return c - 'a'; }
	void insert(char *s,int v)
	{
		int u = 0, n = strlen(s);
		for(int i = 0; i < n; i++)
		{
			int c = idx(s[i]);
			if(!ch[u][c])
			{
				memset(ch[sz],0,sizeof(ch[sz]));
				val[sz] = 0;
				ch[u][c] = sz++;
			}
			u = ch[u][c];
		}
		val[u] ++;
	}

	void getFail()
	{
		queue <int> q;
		f[0] = 0;
		for(int c = 0; c < 26; c++)
		{
			int u = ch[0][c];
			if(u) { f[u] = 0; q.push(u); last[u] = 0; }
		}

		while(!q.empty())
		{
			int r = q.front(); q.pop();
			for(int c = 0; c < 26; c++)
			{
				int u = ch[r][c];
				if(u)
				{
					q.push(u);
					int v = f[r];
					while(v && !ch[v][c]) v = f[v];
					f[u] = ch[v][c];
					last[u] = val[f[u]] ? f[u] : last[f[u]];
				}
				else ch[r][c] = ch[f[r]][c];
			}
		}
	}

	int find(char *s)
	{
		int n = strlen(s);
		int ans = 0;
		int j = 0;
		for(int i = 0; i < n; i++)
		{
			int c = idx(s[i]);
			while(j && !ch[j][c]) j = f[j];
			j = ch[j][c];
			if(val[j]) ans += print(j);
			else ans += print(last[j]);
		}
		return ans;
	}

	int print(int j)
	{
		int ans = 0;
		if(j)
		{
			if(val[j]){ ans += val[j]; val[j] = 0; }
			ans += print(last[j]);
		}
		return ans;
	}
}ac;
char s[1000010];
char tmp[100005];
int main()
{
	freopen("input.txt", "r", stdin);
	int n;
	while(scanf("%d",&n) != EOF)
	{
		ac.init();
		for(int i = 0; i < n; i++) // 读入模式串
		{
			scanf("%s",tmp);
			ac.insert(tmp,i); // 插入模式串
		}
		ac.getFail();
		scanf("%s",s); // 读入原串
		int ans = ac.find(s); // 用原串在模式串中进行匹配
		if(ans != 0) printf("YES\n");
		else printf("NO\n");
	}
	return 0;
}

```

```c++
// 需要输出匹配到的模式串编号
#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <queue>
#include <vector>

using namespace std;

const int maxn = 1000010; // Trie图最大节点个数

struct AC
{	
	int last[maxn],f[maxn];
	int ch[maxn][26];
	int val[maxn];
	int sz;
	void init()
	{
		sz = 1; memset(ch[0],0,sizeof(ch[0]));
	}
	int idx(char c) { return c - 'a'; }

	void insert(char *s,int v) // 插入模式串，这里v是插入模式串编号，可根据题目要求进行修改
	{
		int u = 0, n = strlen(s);
		for(int i = 0; i < n; i++)
		{
			int c = idx(s[i]);
			if(!ch[u][c])
			{
				memset(ch[sz],0,sizeof(ch[sz]));
				val[sz] = 0;
				ch[u][c] = sz++;
			}
			u = ch[u][c];
		}
		val[u] = v;
	}

	void getFail() // 构造失配边
	{
		queue <int> q;
		f[0] = 0;
		for(int c = 0; c < 26; c++)
		{
			int u = ch[0][c];
			if(u) { f[u] = 0; q.push(u); last[u] = 0; }
		}

		while(!q.empty())
		{
			int r = q.front(); q.pop();
			for(int c = 0; c < 26; c++)
			{
				int u = ch[r][c];
				if(u)
				{
					q.push(u);
					int v = f[r];
					while(v && !ch[v][c]) v = f[v];
					f[u] = ch[v][c];
					last[u] = val[f[u]] ? f[u] : last[f[u]];
				}
				else ch[r][c] = ch[f[r]][c];
			}
		}
	}

	int find(char *s)
	{
		int n = strlen(s);
		int ans = 0;
		int j = 0;
		for(int i = 0; i < n; i++)
		{
			int c = idx(s[i]);
			while(j && !ch[j][c]) j = f[j];
			j = ch[j][c];
			if(val[j]) ans += print(j);
			else ans += print(last[j]);
		}
		return ans;
	}

	int print(int j)
	{
		int ans = 0;
		if(j)
		{
			if(val[j]){
				printf("find %d\n", val[j]); // 找到的模式串的编号就是val
				ans++; 
				val[j] = 0; 
			}
			ans += print(last[j]);
		}
		return ans;
	}
}ac;
char s[1000010];
char tmp[100005];
int main()
{
	freopen("input.txt", "r", stdin);
	int n;
	while(scanf("%d",&n) != EOF)
	{
		ac.init();
		for(int i = 0; i < n; i++)
		{
			scanf("%s",tmp);
			ac.insert(tmp,i + 1);  // 将模式串的id值当做value，这样可以查询出现模式串的编号
			cout << tmp << endl;
		}
		ac.getFail();
		scanf("%s",s);
		cout << s << endl;
		int ans = ac.find(s);
		cout << ans << endl;
		if(ans != 0) printf("YES\n");
		else printf("NO\n");
	}
	return 0;
}
```
## 六.STL 工具

### 6.1 pair 

1. 成员变量

   - first
   - second

2. 成员函数

3. 常用操作

   ```c++
   #define pa pair<int,int>
   priority_queue<pa,vector<pa>,greater<pa> >q;
   q.push(make_pair(a, b)); // a代表pair的first， b代表pair的second
   ```

### 6.2 vector

1. 成员函数
   * size()
   * push_back()
   * empty()
   * pop_back()
2. 成员变量
3. 常用
   * sort(nums.begin(), nums.end());

### 6.3 queue
1. 成员函数
	* empty()
	* size()
	* front()
	* back()
	* push()
	* pop()

### 6.4 priority_queue 优先队列

1. 成员函数
	* empty()
	* size()
	* top()
	* push()
	* pop()
	
2. 基本数据类型优先级设置

    ```
    // 以下两种定义方式是等价的, 默认是以大数优先级高，排在前面
    priority_queue<int> q;
    priority_queue<int,vector<int>,less<int> >;//后面有一个空格   
    ```
    
    其中第二个参数( vector )，是来承载底层数据结构堆的容器，第三个参数( less )，则是一个比较类，less 表示数字大的优先级高，而 greater 表示数字小的优先级高。

    如果想让优先队列总是把最小的元素放在队首，只需进行如下的定义：
    
    ```
     priority_queue<int,vector<int>,greater<int> >q;
    ```

3. 结构体优先级设置
    可以在结构体内部重载 ‘<’，改变小于号的功能

    ```c++
        struct student{
            int grade;
            string name;
    
            //重载运算符，grade 值高的优先级大
            friend operator < (student s1,student s2){
                return s1.grade < s2.grade; // 用sturdent的grade值来定义大小关系，默认大者优先
            }
        };
    ```
    eg: 给定n个数对，每个数对由两个数x和y组成，请找出所有数对中x最小的，在x相等的情况下y最小

## 八.常用数据结构

### 8.1树状数组

```

```


### 8.2单点修改线段树
```
#include <iostream>
#include <cstdio>
#include <cstring>
using namespace std;
const int maxn = 50000 + 10;
int C[4 * maxn], n;

// p是希望进行修改的位置，L、R是当前o的区间范围，v是希望在p位置加上的值
void update(int o, int p, int L, int R, int v){
	if(L == R){
		C[o] += v;
	}
	else{
		int M = L + (R - L) / 2;
		if(p <= M) update(o << 1, p, L, M, v);
		else update(o << 1 | 1, p, M + 1, R, v);
		C[o] = C[o << 1] + C[o << 1 | 1];
	}
}
//l、r是询问区间，L、R是o代表的区间范围
int query(int o, int l, int r, int L, int R){
	if(l <= L && r >= R){
		return C[o];
	}
	else{
		int M = L + (R - L) / 2;
		int sum = 0;
		if(l <= M) sum += query(o << 1, l, r, L, M);
		if(r > M) sum += query(o << 1 | 1, l, r , M + 1, R);
		return sum;
	}
}
int main(){
	freopen("input.txt", "r", stdin);
	int T;
	scanf("%d", &T);
	for(int t = 1; t <= T; t++){
		memset(C, 0, sizeof(C));
		scanf("%d", &n);
		for(int i = 1; i <= n; i++){
			int temp;
			scanf("%d", &temp);
			cout << temp << endl;
			update(1, i, 1, n, temp);
		}
		printf("Case %d:\n", t);
		string cmd;
		while(cin >> cmd){
			if(cmd == "End") break;
			else if(cmd == "Query"){
				int l, r;
				scanf("%d%d", &l, &r);
				printf("%d\n", query(1, l, r, 1, n));
			}
			else if(cmd == "Add"){
				int pos, v;
				scanf("%d%d", &pos, &v);
				update(1, pos, 1, n, v);
			}
			else{
				int pos, v;
				scanf("%d%d", &pos, &v);
				update(1, pos, 1, n, -1 * v);
			}
		}
	}
	return 0;
}
```
### 8.3区间修改线段树

### 8.4RMQ-ST表
1. 主要思想：
    * 定义：f[i][j]表示i到i+2^j-1这段区间(长度为2^j)的最大值。

    * 预处理：f[i][0]=a[i]。即i到i区间的最大值就是a[i]。

    * 状态转移：将f[i][j]平均分成两段，一段为f[i][j-1](长度为2^(j-1))，另一段为f[i+2^(j-1)][j-1]长度也为2^(j-1)。

    * 两段的长度均为2^(j-1)。f[i][j]的最大值即这两段的最大值中的最大值。

    * 得到f[i][j]=max(f[i][j-1],f[i+2^(j-1)][j-1])。
    
    * 查询：需要查询的区间为[i,j]，则需要找到两个覆盖这个闭区间的最小幂区间。这两个区间可以重复，因为两个区间是否相交对区间最值没有影响。（如下图）
    ![图](/Users/zhangjiatao/ACM/ACMNote/2.jpg)
    
2. 限制条件
    * 不支持动态更新
    * 不支持区间和询问
    

```
#include <iostream>
#include <cstring>
#include <cstdio>
#include <cmath>
using namespace std;
const int maxn = 1e7 + 10;
int n, q, A[maxn], f[maxn][21];

void RMQ(int N){
    for(int j=1;j<=20;j++)    
        for(int i=1;i<=N;i++)
            if(i+(1<<j)-1<=N) // i + (1<<j)-1 需要减1的意义是保证该区间长度为2^j
                f[i][j]=min(f[i][j-1],f[i+(1<<(j-1))][j-1]); 
}

int main(){
	//freopen("input.txt", "r", stdin);
	while(scanf("%d", &n) != EOF){
		for(int i = 1; i <= n; i++) scanf("%d", &A[i]);
		for(int i = 1; i <= n; i++) f[i][0] = A[i];
		RMQ(n);
		scanf("%d\n", &q);
		int l, r;
		for(int i = 1; i <= q; i++){
			scanf("%d%d", &l, &r);
			int k = (int)(log((double)(r-l+1)) / log(2.0));
			int ans = min(f[l][k], f[r -(1<<k)+1][k]);
			printf("%d\n", ans);
		}
	}
	return 0;
}
```

## 九.dp

### 9.1 01背包问题
1. 问题描述：有N个物品，每个物品只有一件，有两个属性，一个是价值value，一个是花费need，现在总共有m元钱，希望得到的受益最大。


    ```
    #include <iostream>
    #include <cstring>
    #include <cstdio>
    using namespace std;
    const int maxn = 500 + 10;
    const int maxm = 1e6 + 10;
    int n, m;
    int f[maxm], need[maxn], value[maxn];
    int main(){
    //freopen("input.txt", "r", stdin);
    while(scanf("%d%d", &n, &m) != EOF){
    	for(int i = 1; i <= n; i++){
    		scanf("%d%d", &need[i], &value[i]);
    	}
    	memset(f, 0, sizeof(f));
    	for(int i = 1; i <= n; i++){
    		for(int j = m; j >= 0; j--){
    			if(j >= need[i]){
    				f[j] = max(f[j - need[i]] + value[i], f[j]);
    			}else{
    				f[j] = f[j];
    			}
    		}
    	}
    	printf("%d\n", f[m]);
    }
    return 0;
    }
    ```

### 9.2完全背包问题
1.题意：01背包的升级版，每个物品可以取多种
2.要点： 

```
#include <iostream>
#include <cstring>
#include <cstdio>
using namespace std;
const int maxn = 500 + 10;
const int maxm = 1e6 + 10;
int dp[maxm];
int n, m, need[maxn], value[maxn];
int main(){
	//freopen("input.txt", "r", stdin);
	while(scanf("%d%d", &n, &m) != EOF){
		for(int i = 1; i <= n; i++){
			scanf("%d%d", &need[i], &value[i]);
		}
		memset(dp, 0, sizeof(dp));
		for(int i = 1; i <= n; i++){
			for(int j = 0; j <= m; j++){ //注意这里，01背包如果要进行空间优化，必须要从m向0进行计算，而完全背包则必须顺序进行计算。
				if(need[i] > j){
					dp[j] = dp[j];
				}else{
					dp[j] = max(dp[j], dp[j - need[i]] + value[i]);
				}
			}
		}
		printf("%d\n", dp[m]);
	}
	return 0;
}
```
### 经典状态定义（树类）

## 十.计数类问题

### 10.1区间倍数计数
1. 问题描述：$[a, b]$这个区间中有多少个数可以整除K？（这个区间中有多少个数是K的倍数）
2. 解决方法：我们知道$[1, x]$,这个区间中有$\left\lfloor \frac{x}{K} \right\rfloor$个K的倍数，所以$[a, b]$中K的倍数就是$[1, b]-[1,a)$即为:$\left\lfloor \frac{b}{K} \right\rfloor -\left\lfloor \frac{a-1}{K} \right\rfloor$

## 十一.排列组合
### 11.1常用组合数公式
1. $ C^{k}_{n} = \frac{n!}{k!(n-k)!}=\frac{n(n-1)(n - 2)...(n - k+1) }{k!} $ , $A_{n}^{k}= \frac{n!}{(n-k)!}=n(n-1)(n - 2)...(n - k+1)$

* $C^{k}_{n}$ 计算函数（注意，如果需要取模的话，除法部分要进行逆元运算）
```
int myc(int n, int k){
    int ans = 1;
    for(int i = 1; i <= k; i++)
        ans = ans * (n + 1 - i) / i;
    return ans;
}
```
* $A^{k}_{n}$ 如果需要取模的话注意取模操作
```
int mya(int n, int k){
    int ans = 1;
    for(int i = 0; i < k; i++)
        ans = ans * (n - i);
}
```
2. $C^{k}_{n} = C^{k}_{n-1} + C^{k-1}_{n-1} $ 这个公式可以从杨辉三角角度去认识
3. $(x + y)^n = C^{0}_{n}x^0y^n + C^{1}_{n}x^1y^{n-1}... C^{k}_{n}x^ky^n...+C^{n}_{n}x^ny^{0}$
4. 由上面公式可以推导$2^n =C^{0}_{n} + C^{1}_{n}... C^{k}_{n}...+C^{n}_{n} $

### 11.2 特殊排列组合
1. 可重复排列
    * 问题：从n个物体，允许重复的选取k个物体，排成一行，有多少种方法
    * 解析：很简单，k个位置，每个位置有n个选项可选，共有$n^k$
2. 可重复组合
    * 问题：从n个物体，允许重复的选取k个物体，不考虑其次序排成一组，有多少种方法
    * 解析：$C^{k}_{n+r-1}$
3. 不全相异的全排列
    * 问题：有r中物品，第i中物品的数量是$n_i$，$n = n_1+n_2...+n_r$，其中一个种类的物品是没有差别的，将其全部排列成一排，有多少种排列方法
    * 解析：$\frac{n!}{(n_1!*n_2!*n_3!...n_k!)}$

## 其他
### x.1 $\sum _{ k\le i\le n }{ { C }_{ n }^{ i } } $ 递推求解
1. 思路：首先明确$\sum _{ k\le i\le n }{ { C }_{ n }^{ i } } $这个问题是没法一个公式直接O(1)进行求解的，只能O(n)的进行求解。很简单就可以知道$ (1 + 1)^{n} = \sum_{0\le i \le n}{{ C }_{ n }^{ i }} $, 所以我们可以先求解 $ \sum_{0 \le i \le k-1 }{{ C }_{ n }^{ i }} $。这里我们通过递推来进行求解$C_{n}^{i} = C_{n}^{i - 1} * \frac {n - i + 1 }{ i }$

```
#include <iostream>
#include <cstring>
#include <cstdio>
using namespace std;

typedef long long ll;

const ll mod = 1000000007;

ll extgcd(ll a, ll b, ll &x, ll &y){
 ll d = a;
	if(b != 0){
		d = extgcd(b, a % b, y, x);
		y -= (a / b) * x;
	} else {
		x = 1;
		y = 0;
	}
	return d;
}
ll inv (ll a, ll m){
	ll x, y;
	extgcd(a, m, x, y);
	return (m + x % m) % m;
}

ll quick_mod(ll a, ll x, ll m){
	if(x == 0) return 1;
	if(x % 2){
		ll tmp = quick_mod(a, x / 2, m) % m;
		ll ans = (((tmp * tmp) % m) * a) % m;
		return ans;
	} else {
		ll tmp = quick_mod(a, x / 2, m) % m;
		ll ans = (((tmp * tmp) % m) * a) % m;
		return ans;
	}
}

ll n, k;
int main(){
	//freopen("input.txt", "r", stdin);
 	int T;
	scanf("%d", &T);
	for(int t = 1; t <= T; t++){
		scanf("%lld%lld", &n, &k);
		ll now = n % mod; //(C_n^1)
		ll ans = now; // sum 
		for(int i = 2; i <= k - 1; i++){
			ll tmp = inv(i, mod);
			now = (((now * (n - i + 1)) % mod) * tmp) % mod;
			ans += now;
			ans = ans % mod;
		}
		ll all = quick_mod(2, n, mod) - 1;
		ans = (((all - ans) % mod) + mod) % mod ; // 减法取模
		printf("Case #%d: %lld\n", t, ans);

	}
	return 0;
}

```


###x.2 集合运算and数位表示

1. 常用集合表示(注意，以下集合元素均从0号元素开始)
    * 空集:0
    * 只含有第i号元素集合(0 base下面相同){i}: 1 << i
    * 含有全部n个元素集合{0, 1, 2, 3, .. , n - 1 }(其中数字代表元素编号): (1 << n) - 1
    * 判断第i个元素是否属于集合S：if(S>>i&1)
    * 向集合中添加第i个元素$S \cup {\{i}\}$: S | (1 << i)
    * 从集合中除去第i个元素$S \cap {\{i}\}$: S & ~(1 << i)
    * 集合求并$S \cup T$: S | T
    * 集合求交$S \cap T$：S & T
2. 子集枚举算法
    * 枚举n个元素集合{0, 1, 2, 3, .. , n - 1 }(其中数字代表元素编号)的子集

```
for(int s = 0; s <= (1 << n) - 1; s++){
    
}
```

* 枚举集合sup子集，例如sup为01101011, 讲解详见《挑战程序设计竞赛》p157

```
int sub = sup;
do{
    sub = (sub  - 1) & sup
}while(sub != sup)
```

* 枚举n个元素集合{0, 1, 2, 3, .. , n - 1 }(其中数字代表元素编号)的所有大小为K的子集

3. 其他常用数位表示
    * 取最低位1: x & (-x) , 例如x的二进制表示为1100100，那么x & (-x)就是0000100

### x.3 计算多区间相交长度
* 问题描述：给定2个区间[L1,R1],[L2,R2]或给定三个区间[L1,R1],[L2,R2],[L3,R3]求其相交区间长度
* 解决：对于多个区间相交长度，求解其maxl和minr，如果minr>=maxl那么区间[maxl, minr]就是相交区域，否则则不相交

## 常用公式

### 计算几何
### 梯形面积计算
* 已知四边求解面积
$$ \frac{(a + c)}{4 * (a - c)} * sqrt((a + b - c + d) *  (a - b - c + d) *  (a + b - c - d) *  (b - a + c + d) $$

* ps：如何从一个公式判定是否有解，比如这题，面积一定为正，如果为负一定不正确，根号里面不可为负这种条件
  
* 源码

```

#include <iostream>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <cmath>
using namespace std;
const int maxn = 4;
double A[maxn];

// 判定是否合法
int  judge(double a, double b, double  c, double d){
	int flag = 1;
	if(a - c < 1e-8 || (a + b - c + d) * (a - b - c + d) * (a + b - c - d) * (b - a + c + d ) < 1e-8) flag = 0;
	return flag;
}

int main(){
	//freopen("input.txt", "r", stdin);
	int T;
	scanf("%d", &T);	
	while(T--){			
		for(int i = 0; i < 4; i++) scanf("%lf", &A[i]);
		double  ans = -1;
		for(int i = 0; i < 4; i++){
			for(int j = 0; j < 4; j++){
				for(int k = 0; k < 4; k++){
					for(int l = 0; l < 4; l++){
						if(i != j && i != k && i != l && j != k && j != l && k != l){
							double  a = A[i], b = A[j], c = A[k], d = A[l];
							if(judge(a, b, c, d)){
								double tmp = ((a + c) / (4 * (a - c))) * sqrt((a + b - c + d) *  (a - b - c + d) *  (a + b - c - d) *  (b - a + c + d));
								ans = max(ans, tmp);
							}
						}
					}
				}
			}
		}
		
		if(ans + 1 < 1e-8) printf("IMPOSSIBLE\n");
		else printf("%.2lf\n", ans);
	}
}

```

## 日期相关计算
### 由年月日计算周几
```
#include <iostream>
#include <cstdio>
#include <cstring>
#include <map>
using namespace std;

string mp[8];

int Change(int year, int month, int day)  //根据日期判断出星期几  
{  
    if(month == 1 || month == 2)  
    {  
        month += 12;  
        year--;  
    }  
    int c = year / 100;  
    int y = year % 100;  
    int m = month;  
    int d = day;  
    int W = c / 4 - 2 * c + y + y / 4 + 26 * (m + 1) / 10 + d - 1;  
  
    int ans;  
  
    if(W < 0)  
        ans = (W + (-W / 7 + 1) * 7) % 7;  
    else  
        ans = W % 7;  
    if(ans == 0)  
        return ans + 7;  
    return ans;  
}  

int main(){
	mp[1] = "Monday";
	mp[2] = "Tuesday";
	mp[3] = "Wednesday";
	mp[4] = "Thursday";
	mp[5] = "Friday";
	mp[6] = "Saturday";
	mp[7] = "Sunday";
	//freopen("input.txt", "r", stdin);

	int year, month, day;
	while(scanf("%d%d%d", &year, &month, &day) != EOF){
		cout << mp[Change(year, month, day)] << endl;;
		//cout << mp[W] << endl;
	}
	return 0;
}

```
## C++常用基础知识

### 输入输出

* 整行读入

  ```
  	string tmp;
  	while(getline(cin, tmp)){ // 读到文件末尾
  		cout << tmp << endl;
  	}
  ```

* 





### 数据类型相关
* 1 范围

float和double的范围是由指数的位数来决定的。

float的指数位有8位，而double的指数位有11位，分布如下：

float：

1bit（符号位）
8bits（指数位）
23bits（尾数位）
double：

1bit（符号位）
11bits（指数位）
52bits（尾数位）
于是，float的指数范围为-127~+128，而double的指数范围为-1023~+1024，并且指数位是按补码的形式来划分的。其中负指数决定了浮点数所能表达的绝对值最小的非零数；而正指数决定了浮点数所能表达的绝对值最大的数，也即决定了浮点数的取值范围。

float的范围为-2^128 ~ +2^128，也即-3.40E+38 ~ +3.40E+38；double的范围为-2^1024 ~ +2^1024，也即-1.79E+308 ~ +1.79E+308。

* 2 精度

float和double的精度是由尾数的位数来决定的。浮点数在内存中是按科学计数法来存储的，其整数部分始终是一个隐含着的“1”，由于它是不变的，故不能对精度造成影响。

float：2^23 = 8388608，一共七位，这意味着最多能有7位有效数字，但绝对能保证的为6位，也即float的精度为6~7位有效数字；
double：2^52 = 4503599627370496，一共16位，同理，double的精度为15~16位。

### 字符串相关
* strstr(str1, str2): str1与str2均为char[]类型，作用是查找str2在str1中的位置，返回值是char *


