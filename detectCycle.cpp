

struct ListNode
{
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};

ListNode* detectCycle(ListNode* head)
{
    if (head == nullptr || head->next == nullptr) return nullptr;
    
    ListNode* slow = head;
    ListNode* fast = head;

    bool hasCycle = false;
    while (fast != nullptr && fast->next != nullptr)
    {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast)
        {
            hasCycle = true;
            break;
        }
    }
}