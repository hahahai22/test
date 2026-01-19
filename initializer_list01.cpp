
#include <initializer_list>
#include <iostream>

void print_list(std::initializer_list<int> list)
{
    for(int x : list)
    {
        std::cout << x << " ";
    }
}

int main(int argc, char const* argv[])
{
    print_list({1, 4, 3});
    return 0;
}


indices_count_kernel.comp_options = KernelBuildParameters{
    {"VEC_SIZE", vec_size},
    {"TOKENS_PER_CTA", tokens_per_cta},
    {"SMEM_SIZE", smem}}.GenerateFor(kbp::HIP{});


KernelBuildParameters(const std::initializer_list<KBPInit>& defines_)
{
    options.reserve(defines_.size());
    for(const auto& define : defines_)
    {
        assert(ValidateUniqueness(define.data.name));
        options.push_back(define.data);
    }
}

template<class _E>
class initializer_list
{
public:
  typedef _E 		value_type;
  typedef const _E& 	reference;
  typedef const _E& 	const_reference;
  typedef size_t 		size_type;
  typedef const _E* 	iterator;
  typedef const _E* 	const_iterator;

private:
  iterator			_M_array;   // 指向初始化元素的指针
  size_type			_M_len;     // 元素个数

  // The compiler can call a private constructor.
  constexpr initializer_list(const_iterator __a, size_type __l)
  : _M_array(__a), _M_len(__l) { }

public:
  constexpr initializer_list() noexcept
  : _M_array(0), _M_len(0) { }

  // Number of elements.
  constexpr size_type
  size() const noexcept { return _M_len; }

  // First element.
  constexpr const_iterator
  begin() const noexcept { return _M_array; }

  // One past the last element.
  constexpr const_iterator
  end() const noexcept { return begin() + size(); }
};


// ------------------------------------------------------------

auto tensor_size_bytes = [](std::initializer_list<size_t> shape, size_t dtype_size)
{
    size_t num_elems = 1;
    for (auto dim : shape)
        num_elems *= dim;
    return num_elems * dtype_size;
};

size_t scatter_tokens_bytes = tensor_size_bytes({(size_t)real_scatter_tokens, (size_t)hidden_size}, sizeof(hip_bfloat16));
