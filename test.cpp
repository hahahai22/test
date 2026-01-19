#include <iostream>

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr){}
};

ListNode *reverseLinkedList(ListNode *head)
{
    ListNode *previous = nullptr;
    ListNode *current = head;
    while (current != nullptr)
    {
        ListNode *tempNext = current->next;
        current->next = previous;   // 指针反转

        previous = current;
        current = tempNext;
    }
    return previous;
}

ListNode *createList()
{
    ListNode *head = new ListNode(31);
    head->next = new ListNode(32);
    head->next->next = new ListNode(77);
    head->next->next->next = new ListNode(11);
    return head;
}

void printList(ListNode *head)
{
    while (head != nullptr)
    {
        std::cout << head->val << " "
        head = head->next;
    }
    std::cout << std::endl;
    
}

int main()
{

}


