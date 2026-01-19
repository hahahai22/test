
#include <iostream>
#include <vector>

void merge(std::vector<int> &nums1, int m, std::vector<int> &nums2, int n)
{
    int i = m - 1, j = n - 1, k = m + n - 1;
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
    std::vector<int> nums1 = {0, 1, 2, 3, 0, 0, 0, 0, 0};
    int m = 4;
    std::vector<int> nums2 = {32, 34, 42, 89, 99};
    int n = 5;

    merge(nums1, m, nums2, n);

    for (int num : nums1)
    {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}
