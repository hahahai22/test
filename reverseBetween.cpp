#include <iostream>

struct ListNode
{
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};

class Solution
{
public:
    ListNode *reverseBetween(ListNode *head, int m, int n)
    {
        ListNode *current = head;
        ListNode *tempNodem = nullptr;
        ListNode *tempNoden = nullptr;
        int i = 0;
        while (head != nullptr)
        {
            current = current->next;
            i++;
            if (i == (m - 1))
            {
                tempNodem = current;
            }

            if (i == n)
            {
                tempNoden = current;
            }
        }
        ListNode *reverseCurrent = tempNodem;
        ListNode *previous = nullptr;
        while (reverseCurrent != tempNoden)
        {
            ListNode *reverseTempNode = reverseCurrent->next;
            reverseCurrent->next = previous;

            previous = reverseCurrent;
            reverseCurrent = reverseTempNode;
        }
        tempNodem->next = previous;
        return head;
    }
};

// class Solution
// {
// public:
//     ListNode *reverseBetween(ListNode *head, int m, int n)
//     {
//         // write code here
//         if (head == nullptr || m == n)
//             return head;

//         ListNode dummy(0);
//         dummy.next = head;
//         ListNode *prev = &dummy;

//         for (int i = 1; i < m; ++i)
//         {
//             prev = prev->next;
//         }

//         ListNode *current = prev->next;
//         ListNode *nextNode;

//         for (int i = 0; i < n - m; ++i)
//         {
//             nextNode = current->next;
//             current->next = nextNode->next;
//             nextNode->next = prev->next;
//             prev->next = nextNode;
//         }

//         return dummy.next;
//     }
// };

// 打印链表
void printList(ListNode *head)
{
    while (head != nullptr)
    {
        std::cout << head->val << " ";
        head = head->next;
    }
    std::cout << std::endl;
}

// 创建链表
ListNode *createList()
{
    ListNode *head = new ListNode(1);
    head->next = new ListNode(2);
    head->next->next = new ListNode(3);
    head->next->next->next = new ListNode(4);
    head->next->next->next->next = new ListNode(5);
    return head;
}

int main()
{
    ListNode *head = createList();
    printList(head); // 输出: 1 2 3 4 5

    Solution solution;
    head = solution.reverseBetween(head, 2, 4);
    printList(head); // 输出: 1 4 3 2 5

    return 0;
}
