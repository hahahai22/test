#include <iostream>
#include <vector>
#include <memory>

class Edge;
class Node;

class Node
{
public:
    std::vector<std::shared_ptr<Edge>> in_edge_;
    std::vector<std::shared_ptr<Edge>> out_edge_;

    inline void addInEdge(std::shared_ptr<Edge> edge)
    {
        in_edge_.emplace_back(edge);
    }

    inline void addOutEdge(std::shared_ptr<Edge> edge)
    {
        out_edge_.emplace_back(edge);
    }
};

class Edge
{
public:
    Edge(std::weak_ptr<Node> a, std::weak_ptr<Node> b) : begin_node(a), end_node(b)
    {
    }

    std::weak_ptr<Node> begin_node;
    std::weak_ptr<Node> end_node;
};

int main(int argc, const char* argv[])
{
    auto node1 = std::make_shared<Node>();
    auto node2 = std::make_shared<Node>();

    auto edge = std::make_shared<Edge>(node1, node2);

    node1->addOutEdge(edge);
    node2->addInEdge(edge);

    std::cout << "节点1的边数量: " << node1->out_edge_.size() << std::endl;
    std::cout << "节点2的边数量: " << node2->in_edge_.size() << std::endl;
}