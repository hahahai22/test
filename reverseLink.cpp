#include <iostream>
using namespace std;

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
        ListNode *nextTemp = current->next; // 解引用current，获取current指向的ListNode对象，访问ListNode对象的next成员（next是指针）。
        current->next = previous;           // 实质是指针指向反转
        previous = current;
        current = nextTemp;
    }
    return previous;
}

void printList(ListNode *head)
{
    while (head != nullptr)
    {
        std::cout << head->val << " ";
        head = head->next;
    }
    cout << endl;
}

ListNode *createLinkedList()
{
    ListNode *head = new ListNode(1);
    head->next = new ListNode(2);
    head->next->next = new ListNode(3);
    head->next->next->next = new ListNode(7);
    return head;
}

int main()
{
    ListNode *head = createLinkedList();
    printList(head);

    ListNode *reverseHead = reverseLinkedList(head);
    printList(reverseHead);
}
