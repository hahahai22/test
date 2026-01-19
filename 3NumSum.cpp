#include <vector>
#include <algorithm>
using namespace std;

vector<vector<int>> getSolution(vector<int>& nums)
{
    vector<vector<int>> result;

    int n = nums.size();
    sort(nums.begin(), nums.end());

    for(int i = 0; i < n - 2; ++i)
    {
        int j = i + 1;
        int k = n - 1;

        int target = -nums[i];
        while(j < k)

    }
}