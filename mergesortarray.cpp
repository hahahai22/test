#include <iostream>
#include <vector>

void mergeSortedArray(std::vector<int> &nums1, int m, std::vector<int> &nums2, int n)
{
    int i = m - 1;
    int j = n - 1;
    int k = m + n - 1;
    while (i >= 0 && j >= 0)
    {
        if (nums1[i] > nums2[j])
        {
            nums1[k] = nums1[i];
            --i;
        }
        else
        {
            nums1[k] = nums2[j];
            --j;
        }
        --k;
    }

    while (j >= 0)
    {
        nums1[k] = nums2[j];
        --j;
        --k;
    }
}

int main()
{
    std::vector nums1 = {1, 2, 3, 0, 0, 0};
    int m = 3;
    std::vector nums2 = {2, 5, 6, 8};
    int n = 4;

    mergeSortedArray(nums1, m, nums2, n);
    for (int num : nums1)
    {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    return 0;
}
