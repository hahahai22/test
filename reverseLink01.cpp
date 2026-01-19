#include <iostream>

struct ListNode
{
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};

ListNode *reverseLinkedList(ListNode *head)
{
    ListNode *previous = nullptr;
    ListNode *current = head;
    while (current != nullptr)
    {
        ListNode *nextTemp = current->next;
        current->next = previous;
        previous = current;
        current = nextTemp;
    }
    return previous;
}

ListNode *creatLinkedList()
{
    ListNode *head = new ListNode(22);
    head->next = new ListNode(11);
    head->next->next = new ListNode(13);
    head->next->next->next = new ListNode(17);
    return head;
}

void printList(ListNode *head)
{
    while(head != nullptr)
    {
        std::cout << head->val << " ";
        head = head->next;
    }
    std::cout << std::endl;
}

int main()
{
    ListNode *head = creatLinkedList();
    printList(head);
    ListNode *reverseHead = reverseLinkedList(head);
    printList(reverseHead);
}
