#include <iostream>
#include <vector>
#include <memory>

// 修复点
class Edge;
class Node;

class Node
{
public:
    std::vector<std::shared_ptr<Edge>> in_edges_;
    std::vector<std::shared_ptr<Edge>> out_edges_;

    inline void addInEdge(std::shared_ptr<Edge> edge) { in_edges_.emplace_back(edge); }

    inline void addOutEdge(std::shared_ptr<Edge> edge) { out_edges_.emplace_back(edge); }
};

class Edge
{
public:
    Edge(std::shared_ptr<Node> a, std::shared_ptr<Node> b) : begin_node_(a), end_node_(b) {}

    // std::shared_ptr<Node> begin_node_;
    // std::shared_ptr<Node> end_node_;
    std::weak_ptr<Node> begin_node_;
    std::weak_ptr<Node> end_node_;
};

int main(int argc, const char* argv[])
{
    // 创建节点1、节点2
    auto node1 = std::make_shared<Node>();
    auto node2 = std::make_shared<Node>();

    // 创建边（从node1指向node2）
    auto edge = std::make_shared<Edge>(node1, node2);

    // 给节点添加边
    node1->addOutEdge(edge); // node1的出边
    node2->addInEdge(edge);  // node2的入边

    std::cout << "节点1的出边数量: " << node1->out_edges_.size() << std::endl;
    std::cout << "节点2的入边数量: " << node2->in_edges_.size() << std::endl;

    return 0;
}
