#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

// #define PT_JIT_PARSER_DEBUG // uncomment this line to enable debug mode
#ifdef PT_JIT_PARSER_DEBUG
/**
 * @brief Print debug message if PT_JIT_PARSER_DEBUG is defined
 * otherwise, this will be optimized off
 */
#define print_debug_msg(...)  \
  {                           \
    std::cout << __VA_ARGS__; \
    std::cout.flush();        \
  }
#else
#define print_debug_msg(...) {}
#endif

void print_hw_result(const char *mark,
                     const char *description,
                     std::vector<std::string> info_list)
{
  auto print_to = [=](std::ostream &o) {
    o << '[' << mark << "] " << description << std::endl;
    for (auto &line: info_list)
      o << "\t" << line << std::endl;
  };

  print_to(std::cout);

  std::ofstream f(std::string("hw") + mark + "-output.txt");
  print_to(f);
}

std::string replace(std::string&& str,
                    const std::string from,
                    const std::string to)
{
  size_t start_pos = str.find(from);
  while (start_pos != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos = str.find(from);
  }

  return str;
}

bool has_keyword(std::string word,
                 std::string key)
{
  auto lower = [](std::string &s) {
    std::transform(s.begin(), s.end(), s.begin(),
      [](unsigned char c){ return std::tolower(c); }); };
  lower(word);
  lower(key);
  
  return word.find(key) != word.npos;
}

namespace cal_mac {

int64_t cal_conv2d_macs(c10::IntArrayRef k_shape,
                        c10::IntArrayRef output_shape,
                        int i_channels,
                        int o_channels,
                        int groups,
                        bool with_bias = false)
{
  if (k_shape.size() != 2) {
    std::stringstream ss;
    ss << "Kernel size should be 2, not " << k_shape.size();
    throw std::invalid_argument(ss.str());
  }
  auto k_ops = (static_cast<double>(k_shape[0]) * k_shape[1] * i_channels) / groups;
  auto o_size = output_shape[0] * output_shape[1] * o_channels;
  return static_cast<int64_t>((k_ops + with_bias) * o_size); // count bias if needed
}

int64_t cal_linear_macs(c10::IValue &i_shape,
                        c10::IValue &o_shape,
                        bool with_bias = false)
{  
  int64_t res = 1;
  for (auto &n: i_shape.toTensor().sizes())
    if (n)
      res *= n;
  int64_t o_res = 1;
  for (auto &n: o_shape.toTensor().sizes())
    if (n)
      o_res *= n;
  res = (res + with_bias) * o_res; // count bias if needed
  return res;
}

enum LayerType { CONV, LINEAR, UNKNOWN };

LayerType layer_type(std::string name) {
  if (has_keyword(name, "conv"))
    return CONV;
  else if (has_keyword(name, "linear") ||
           has_keyword(name, "gemm"))
    return LINEAR;
  return UNKNOWN;
}

};

namespace ptjit_parser {

const char *NAMESPACE = "PytorchJitParser";

using var_t = std::pair<std::string,
                        std::string>;

/**
 * @brief Parser to tokenize given string
 */
class parser {
private:
  const std::string str;
  size_t from;
public:
  parser(std::string str):
    str(std::move(str)), from(0) {}

  /**
   * @brief Consume and return a string till given stopper
   * 
   * @param stopper If it's `nullptr`, return the remaining string.
   * @param ignore 
   * @return std::unique_ptr<std::string> 
   */
  std::unique_ptr<std::string>
  consume(const char *stopper = nullptr,
            int ignore = 0)
  {
    if (stopper == nullptr) {
      auto old_from = from;
      from = str.npos;
      return std::make_unique<std::string>(str.substr(old_from));
    }
    
    auto to = str.find(stopper, from);

    if (to == str.npos) // nothing exists
      return nullptr;
    
    std::string res = str.substr(from + ignore,
                                 to - from - ignore);
    from = to + std::strlen(stopper);
    return std::make_unique<std::string>(res);
  }

  /**
   * @brief Find next string to consume w.r.t. to 
   *        the closest stopper candidate in `candidates`
   * 
   * @param candidates list of stoppers
   * @return `{ found_idx (-1 if DNE), ptr (nullptr if DNE) }`
   */
  std::pair<size_t, std::unique_ptr<std::string>>
    consume_first_from(std::vector<const char*> candidates,
                       size_t ignore = 0)
  {
    size_t min_to = str.npos;
    size_t min_idx;

    for (size_t i = 0; i < candidates.size(); ++i) {
      auto local_to = str.find(candidates[i], from);
      if (local_to != str.npos && local_to < min_to) {
        min_to = local_to;
        min_idx = i;
      }
    }

    if (min_to == str.npos) // nothing exists
      return { -1, nullptr };

    auto to = str.find(candidates[min_idx], from);
    std::string res = str.substr(from + ignore,
                                 to - from - ignore);
    from = to + std::strlen(candidates[min_idx]);

    return { min_idx, std::make_unique<std::string>(res) };
  }
};

/**
 * Convert node name to that of key in node_io_map, nv_map.
 * Note: actually `.X` means "X indentions" in Python code
 */
inline std::string conv_name(const char *nn) {
  std::string name(nn);
  auto cut = name.find('.');
  if (cut == name.npos)
    return name;
  return name.substr(0, cut); 
}

inline std::string conv_name(std::string &nn) {
  return conv_name(nn.c_str());
}

inline std::string conv_name(std::string &&nn) {
  return conv_name(nn.c_str());
}

/**
 * Container of torch function signature
 */
struct fn_t {
  std::string fn_name;
  std::vector<std::string> inputs;
  /**
   * `{ output_name, output_type }`
   */
  var_t output;
  std::string comment;
  /**
   * unique pointer containing `{ option_name, option_value }`
   */
  std::unique_ptr<var_t> option;
  
  fn_t(std::string &fn_name,
       std::vector<std::string> &inputs,
       var_t &&output,
       std::string &comment,
       std::unique_ptr<var_t> option)
       : fn_name(fn_name), inputs(inputs),
         output(std::move(output)), comment(comment),
         option(std::move(option)) {}
         
  fn_t(std::string &fn_name,
       std::vector<std::string> &inputs,
       var_t &output,
       std::string &comment,
       std::unique_ptr<var_t> option)
       : fn_name(fn_name), inputs(inputs),
         output(output), comment(comment),
         option(std::move(option)) {}

  fn_t(std::string &&fn_name,
       std::vector<std::string> &&inputs,
       var_t &&output,
       std::string &&comment,
       std::unique_ptr<var_t> option)
       : fn_name(std::move(fn_name)), inputs(std::move(inputs)),
         output(std::move(output)), comment(std::move(comment)),
         option(std::move(option)) {}
};

using node_io_t = std::unordered_map<std::string,
                                     std::pair<std::vector<std::string>,
                                     std::string>>;

// output-operation map type
using o_op_t = std::unordered_map<std::string,
                                  std::pair<c10::IValue (*)(std::vector<c10::IValue> &),
                                            std::vector<std::string>>>;

// name-value map
using nv_t = std::unordered_map<std::string,
                                c10::IValue>;

/**
 * @brief Parse node which calls some function
 * 
 * Can parse things like:
 * `%OUTPUT_NAME : TYPE = FUNCTION_NAME[OPTION=OPTION_VAL](%P1, %P2, ...) # COMMENTS`
 *
 * @param node torch JIT node
 * @return `{ FUNCTION_NAME, { input_list, output, COMMENTS, [ nullptr | OPTION ] } }`
 *    where input_list: vector of parameter name
 *          output: OUTPUT_NAME : TYPE
 */
fn_t parse_fn_call(torch::jit::Node *node)
{
  // Dump node to string
  std::stringstream ss;
  ss << *node;
  const std::string dump_str = ss.str();
  parser p(dump_str);

  // Get name of output
  // Note: ignore 1 is because
  //    ir of variable prefixed with "$"
  auto output_name = conv_name(std::move(*p.consume(" : ", 1)));
  print_debug_msg("[" << NAMESPACE << "::ParseFnCall] output_name: " << output_name);

  // Get type of output
  auto output_type = std::move(*p.consume(" = "));
  print_debug_msg(", output_type: " << output_type);

  // Get function to be invoked
  auto multi_parse = p.consume_first_from({ "[", "(" });
  std::string invoke = std::move(*multi_parse.second); // func name
  print_debug_msg(", invoke: " << invoke);

  std::unique_ptr<var_t> option = nullptr;
  // Check if option value exists
  if (multi_parse.first == 0) {
    // option value exists
    auto option_name = std::move(*p.consume("="));
    print_debug_msg(", option_name: " << option_name;)
    auto option_value = std::move(*p.consume("]"));
    print_debug_msg(", option_value: " << option_value);
    option = std::make_unique<var_t>(option_name, option_value);
  }

  if (multi_parse.first == 0) {
    print_debug_msg(", option after scope: " << *option);
  }

  // Get input parameters
  std::vector<std::string> params;
  std::unique_ptr<std::string> param = nullptr;
  while (true) {
    // Note: starting index is from + 1 is because
    //    ir of variable prefixed with "$"
    param = p.consume(", ", 1);
    if (param.get() == nullptr)
      break;
    auto p = conv_name(std::move(*param));
    print_debug_msg(", param: " << p);
    params.push_back(std::move(p));
  }

  // Note: starting index is from + 1 is because
  //    ir of variable prefixed with "$"
  param = p.consume(")", 1);
  if (param.get() && !param->empty()) {
    auto p = conv_name(std::move(*param));
    print_debug_msg(", param: " << p);
    params.push_back(std::move(p));
  }

  p.consume(" # ");
  auto comment = std::move(*p.consume());
  print_debug_msg(", comment: " << comment);

  return { invoke,
           params,
           { output_name, output_type },
           comment,
           std::move(option) };
}

/**
 * @brief 
 * @return std::pair<std::string,
 * std::pair<c10::IValue (*)(std::vector<c10::IValue> &),
 * std::vector<std::string>>> 
 */
std::pair<std::string,
          std::pair<c10::IValue (*)(std::vector<c10::IValue> &),
                    std::vector<std::string>>>
parse_subscription(torch::jit::Node *node)
{
  auto fn_info = parse_fn_call(node);
  // make sure that function name is really aten::select
  assert(fn_info.fn_name == "aten::select");

  /**
   * My implementation of `aten::select` function
   * in order to invoke it manually
   */
  static auto my_aten_select = [](std::vector<c10::IValue> &v){
    return c10::IValue(v[0].toTensor()
                           .select(v[1].toInt(),
                                   v[2].toInt()));
  };

  return { fn_info.output.first,
            { my_aten_select,
              fn_info.inputs } };
}

using const_t = std::pair<std::string,
                          c10::IValue>;

/**
 * @brief Parse prim::Constant
 * 
 * @param node 
 * @param debug_mode 
 * @return const_t 
 */
const_t
parse_const(torch::jit::Node *node)
{
  // Parse node
  auto fn_info = parse_fn_call(node);

  // Assure this is really a constant
  assert(fn_info.fn_name == "prim::Constant");

  // construct constant and return
  c10::IValue v;
  if (fn_info.output.second == "int") {
    int64_t _v;
    std::stringstream ss;
    ss << fn_info.option->second;
    ss >> _v;
    v = c10::IValue(_v);
  }

  return { fn_info.output.first, v };
}

/**
 * @brief Parse prim::Constant
 * 
 * @param node 
 * @return const_t 
 */
const_t
parse_list_construct(torch::jit::Node *node,
                     nv_t &nv_map)
{
  // Parse node
  auto fn_info = parse_fn_call(node);

  // Assure this is really a list-construct
  assert(fn_info.fn_name == "prim::ListConstruct");

  // construct constant and return
  int lim = fn_info.inputs.size();
  at::Tensor v = torch::zeros(lim);
  if (fn_info.output.second == "int[]") {
    // print_debug_msg("ListConstruct: [");
    for (int i = 0; i < lim; ++i) {
      v[i] = nv_map[fn_info.inputs[i]].toInt();
      // print_debug_msg(v[i].item().toInt() << ", ");
    }
    // print_debug_msg("]");
  }

  return { fn_info.output.first, v };
}

/**
 * @brief Get all attributes in a module
 * 
 * @param model 
 * @param nv_map 
 */
void get_attributes(torch::jit::Module &model,
                    nv_t &nv_map)
{
  for (auto a: model.named_attributes()) {
    auto name = conv_name(a.name.c_str());
    if (a.value.toIValue().isTensor()) {
      nv_map[name] = a.value;
    
      // std::stringstream _ss;
      // _ss << nv_map[name].toTensor();
      // auto _str = replace(_ss.str(), "\n", ", ");

      // print_debug_msg("\tSet attribute [ "
      //   << name << " ] to << " << _str.substr(0, 30)
      //   << " >>, which is a Tensor" << std::endl);

    } else {
      nv_map[a.name] = a.value;
      // print_debug_msg("\tSet attribute [ " << a.name
      //   << " ] to << " << a.value.toIValue()
      //   << " >>" << std::endl);
    }
  }
}

/**
 * @brief Trace forward function of a model to 
 *    update given maps listed below...
 * 
 * @param model target model
 * @param node_io_map node_name - i0 name map
 * @param oo_map output - operator map
 * @param nv_map name - value map
 */
void trace(torch::jit::script::Module &model,
           node_io_t &node_io_map,
           o_op_t &oo_map,
           nv_t &nv_map
)
{
  auto graph = model.get_method("forward").graph();
  
  // parse IO relationship in all the nodes
  for (auto n: graph->nodes()) {
    switch (n->kind()) {
    case torch::jit::prim::Constant: {
      auto parsed_const = ptjit_parser::parse_const(n);
      nv_map[parsed_const.first] = parsed_const.second;
      break;
    }
    case torch::jit::prim::ListConstruct: {
      auto parsed_const = ptjit_parser::parse_list_construct(n, nv_map);
      nv_map[parsed_const.first] = parsed_const.second;
      break;
    }
    case torch::jit::aten::select: {
      auto fn_info = ptjit_parser::parse_subscription(n);
      oo_map[fn_info.first] = { fn_info.second };
      break;
    }
    case torch::jit::prim::GetAttr: {
      // auto parsed_attr = ptjit_parser::parse_self_attribute(n, nv_map, model);
      // nv_map[parsed_attr.first] = parsed_attr.second;
      break;
    }
    case torch::jit::prim::CallMethod: {
      // auto parsed_attr = ptjit_parser::parse_self_attribute(n, nv_map, model);
      // nv_map[parsed_attr.first] = parsed_attr.second;
      // auto fn_info = parse_fn_call(n);
      // if (fn_info.option.get() && has_keyword(fn_info.option->second, "forward")) {
      //   print_debug_msg("Find tracable function for module: " << *n << std::endl);
      // }
      break;
    }
    default:
      break;
    }

    print_debug_msg("{ " << n->kind().toDisplayString() << " } ");
    auto inputs = n->inputs();
    if (inputs.size() < 2) {
      print_debug_msg("\n");
      continue;
    }
    
    auto node_name = conv_name((*inputs.begin())->debugName());
    auto output_name = conv_name((*n->outputs().begin())->debugName());

    std::vector<std::string> v;
    for (auto &i_info: inputs)
      v.emplace_back(conv_name(i_info->debugName()));

    node_io_map[node_name] = { v, output_name };
    
    // print debug messages
    print_debug_msg("(Node: " << node_name << ")\tInput: ");
    for (auto i: v)
      print_debug_msg(i << ", ");
    print_debug_msg("\t ||\t\tOutputs: " << output_name << ", " << std::endl);
  }
}

/**
 * @brief Get the output name and inputs values for forward method
 *    Has side effect that update oo_map
 *    according to post evaluation
 * 
 * @param model 
 * @param node_io_map 
 * @param oo_map 
 * @param nv_map 
 * @return std::pair<std::string,
 * std::vector<c10::IValue>> 
 */
std::pair<std::string,
          std::vector<c10::IValue>>
get_o_name_and_inputs(torch::jit::Named<torch::jit::Module> &model,
                      node_io_t &node_io_map,
                      o_op_t &oo_map,
                      nv_t &nv_map) {
  /**
   * Collect inputs to forward
   */
  auto io = node_io_map[conv_name(model.name)];
  auto inputs = std::vector<c10::IValue>();
  // Note: i starts from 1 is to skip first input (model itself)
  for (size_t i = 1; i < io.first.size(); ++i) {
    auto val = nv_map[io.first[i]];
    
    print_debug_msg("\tFetched input: " << io.first[i]);

    if (val.tagKind() != "None") {
      print_debug_msg(", " << val.tagKind() << std::endl);
      inputs.push_back(val);
    } else {
      print_debug_msg(", value DNE, check post eval...");
      // check post evaluation w.r.t. this input
      auto entry = oo_map.find(io.first[i]);
      if (entry != oo_map.end()) {
        print_debug_msg(", find pending evaluation, exe it...");
        // do post-evaluation
        auto fn_ptr = entry->second.first;
        std::vector<c10::IValue> post_ev_inputs;
        for (auto &name: entry->second.second)
          post_ev_inputs.push_back(nv_map[name]);
        auto hidden_o = fn_ptr(post_ev_inputs);
        nv_map[io.first[i]] = hidden_o;
        // fetch calculated result as new input
        inputs.push_back(hidden_o);
        print_debug_msg(", evaluation finished, of type " << hidden_o.tagKind() << std::endl);
      }
    }
  }

  return { io.second, inputs };
}


/**
 * Function type which deals with data from each evaluation
 * parameters: `{ output, input_list }`
 */
using io_delegate_fn_t = std::function<void (torch::jit::Named<torch::jit::Module> &,
                                             c10::IValue &,
                                             std::vector<c10::IValue> &)>;

/**
 * @brief Evaluate output data
 * 
 * @param model 
 * @param node_io_map 
 * @param oo_map 
 * @param nv_map 
 * @return name_of_output, you can access `nv_map`
 *         with output name to get its value
 */
void evaluate(torch::jit::script::Module &model,
              c10::IValue &input,
              node_io_t &node_io_map,
              o_op_t &oo_map,
              nv_t &nv_map,
              std::vector<io_delegate_fn_t> missions)
{
  /**
   * Initial dummy input
   */
  nv_map[conv_name("input_1")] = input;
  /**
   * Do inference manually by running forward functions
   *    according to 
   */
  for (auto submodule: model.named_children()) {
    print_debug_msg("(" << submodule.name << ")\n");
    
    /**
     * Collect inputs to forward
     */
    auto io_info = get_o_name_and_inputs(submodule,
                                         node_io_map,
                                         oo_map,
                                         nv_map);
    auto &o_name = io_info.first;
    auto &inputs = io_info.second;

    // record attributes
    get_attributes(submodule.value, nv_map);
    
    // avoid nodes (lines) that doesn't have forward method
    if (!submodule.value.find_method("forward").has_value()) {
      continue;
    }
    
    // forward and collect hidden output info
    auto hidden_o = submodule.value.forward(inputs);

    // add output of hidden layer to tensor map
    nv_map[o_name] = hidden_o;

    // invoke all delegated missions
    for (auto &d: missions)
      d(submodule, hidden_o, io_info.second);
  }
}

}

/**
 * @brief Evaluate output { size, shape } list
 *    and dispatch delegators to `evaluation`
 * 
 * @param model target model
 * @param node_io_map node_name - i0 name map
 * @param oo_map output - operator map
 * @param nv_map name - value map
 * @return std::vector<std::pair<int64_t, torch::IntArrayRef>>
 */
std::pair<std::vector<std::pair<int64_t,
                      torch::IntArrayRef>>,
          std::pair<int64_t,
                    int64_t>>
evaluate_with_missions(torch::jit::script::Module &model,
                       ptjit_parser::node_io_t &node_io_map,
                       ptjit_parser::o_op_t &oo_map,
                       ptjit_parser::nv_t &nv_map)
{
  std::vector<std::pair<int64_t,
                        torch::IntArrayRef>> output_sizes;

  /**
   * @brief Delegator to fetch output size
   * 
   */
  static auto o_size_mission = [&](torch::jit::Named<torch::jit::Module> &submodule,
                                   c10::IValue & o,
                                   std::vector<c10::IValue> & _)
  { 
    // collect layer bandwidth
    auto hidden_o_t = o.toTensor();
    
    // print debug message
    std::stringstream ss;
    ss << "\t[";
    for (int i = 0; i < hidden_o_t.dim(); ++i)
      ss << hidden_o_t.size(i) << (i + 1 == hidden_o_t.dim() ? "" : ", ");
    ss << "] (" << hidden_o_t.element_size() << ")";
    print_debug_msg(ss.str() << std::endl);
    
    output_sizes.push_back({ hidden_o_t.element_size(), hidden_o_t.sizes() });
  };

  int64_t conv_macs = 0;
  std::vector<int64_t> conv_macs_v;
  int64_t gemm_macs = 0;
  static auto mac_cal_mission = [&](torch::jit::Named<torch::jit::Module> &submodule,
                                    c10::IValue & o,
                                    std::vector<c10::IValue> & is)
  {
    
    /**
     * Calculate macs for linear and convolution layers
     */
    switch (cal_mac::layer_type(submodule.name)) {
    case cal_mac::LayerType::CONV: {
      /**
      * @brief Get width and height from a convolution tensor
      *    i.e. get value of the last two indices
      */
      static auto get_conv_tensor_hw = [](c10::IValue &tensor) {
        auto arr = tensor.toTensor().sizes();
        return arr.slice(arr.size() - 2, 2);
      };
      /**
      * @brief Get number of channels from a tensor
      */
      static auto get_conv_tensor_ch = [](c10::IValue &tensor) {
        auto arr = tensor.toTensor().sizes();
        return arr[arr.size() - 3];
      };
      print_debug_msg("Recognized (" << submodule.name << ")\n");
      ptjit_parser::node_io_t loc_node_io_map; // useless here
      ptjit_parser::o_op_t loc_oo_map;
      ptjit_parser::nv_t loc_nv_map;

      ptjit_parser::trace(submodule.value,
                          loc_node_io_map,
                          loc_oo_map,
                          loc_nv_map);

      ptjit_parser::get_attributes(submodule.value, loc_nv_map);

      print_debug_msg("All node io map..." << std::endl);

      int groups = -1;
      c10::IValue kernel;
      for (auto _i: loc_node_io_map) {
        // Note: node of aten::_convolution use 'input' as output name
        if (_i.second.second == "input") {
          kernel = loc_nv_map[_i.second.first[1]];
          groups = loc_nv_map[_i.second.first[8]].toInt();
        }
        print_debug_msg(_i << std::endl);
      }

      print_debug_msg("All name-value map..." << std::endl);
      for (auto _i: loc_nv_map) {
        std::stringstream ss;
        if (_i.second.isTensor()) {
          ss << "(Tensor) [ " << _i.first << " ] with shape: " << _i.second.toTensor().sizes();
        } else {
          ss << _i;
        }
        print_debug_msg(replace(ss.str(), "\n", " ") << std::endl);
      }

      print_debug_msg("Output shape: " << o.toTensor().sizes() << std::endl);
      print_debug_msg("kernel: " << get_conv_tensor_hw(kernel) << std::endl);
      print_debug_msg("groups: " << groups << std::endl);

      print_debug_msg("------------------" << std::endl);

      /**
       * Ref: source code `torch/jit/_shape_functions.py`.
       *
       * ```Python
       * conv(input,
       *      weight,
       *      bias,
       *      stride: IntArrayRef[2],
       *      pad: IntArrayRef[2],
       *      dilation: IntArrayRef[2],
       *      transposed: bool,
       *      output_padding: IntArrayRef[2],
       *      groups: int,
       *      benchmark: bool,
       *      deterministic: bool,
       *      cudnn_enabled: bool,
       *      allow_tf32: bool )
       * ```
       */
      auto loc_macs = cal_mac::cal_conv2d_macs(get_conv_tensor_hw(kernel),
                                               get_conv_tensor_hw(o),
                                               get_conv_tensor_ch(is[0]),
                                               get_conv_tensor_ch(o),
                                               groups,
                                               loc_nv_map.find("bias") != loc_nv_map.end());
      print_debug_msg("Mac of this conv2d layer: " << loc_macs << std::endl);
      conv_macs_v.push_back(loc_macs);
      conv_macs += loc_macs;
      break;
    }
    case cal_mac::LayerType::LINEAR: {
      print_debug_msg("Recognized " << submodule.name << std::endl);
      ptjit_parser::node_io_t loc_node_io_map; // useless here
      ptjit_parser::o_op_t loc_oo_map;
      ptjit_parser::nv_t loc_nv_map;

      ptjit_parser::trace(submodule.value,
                          loc_node_io_map,
                          loc_oo_map,
                          loc_nv_map);

      ptjit_parser::get_attributes(submodule.value, loc_nv_map);
      
      auto loc_macs = cal_mac::cal_linear_macs(is[0],
                                               o,
                                               loc_nv_map.find("bias") != loc_nv_map.end());
      print_debug_msg("Mac of this linear layer: " << loc_macs << std::endl);
      gemm_macs += loc_macs;

      break;
    }
    default:
      break;
    }
  };
  
  /**
   * Initial dummy input
   */
  c10::IValue input = torch::zeros({1, 3, 224, 224});

  ptjit_parser::evaluate(model,
                        input,
                        node_io_map,
                        oo_map,
                        nv_map,
                        { o_size_mission, mac_cal_mission });

  print_debug_msg("Conv macs: " << conv_macs << std::endl);
  for (auto &n: conv_macs_v)
    print_debug_msg(n << std::endl);
  print_debug_msg("Gemm macs: " << gemm_macs << std::endl);

  return { output_sizes, { conv_macs, gemm_macs } };
}

int main() {
  /**
   * Fetch `HOME` environment variable to get model path
   *    for arbitrary user using this Unix-based container
   */
  torch::jit::script::Module model =
      torch::jit::load(std::string(getenv("HOME")) +
        "/projects/lab02/hw2-4/models/model.pt");

  int64_t total_param_bytes = 0;
  for (auto p : model.parameters())
    total_param_bytes += p.numel() * p.element_size();

  /**
   * @brief hw 2-4-1: Calculate model memory requirements for storing weights
   */
  { /* Collect and print data */
    std::stringstream ss;
    ss << "Total memory for parameters: " << total_param_bytes << " bytes";

    print_hw_result("2-4-1",
                    "Calculate model memory requirements for storing weights",
                    {ss.str()});
  }

  /**
   * @brief Dump model info for debugging
   */
  print_hw_result("2-4-0",
                  "Dump model info",
                  { model.dump_to_str(true, false, false).c_str() });

  ptjit_parser::node_io_t node_io_map;
  ptjit_parser::o_op_t oo_map;
  ptjit_parser::nv_t nv_map;

  ptjit_parser::trace(model, node_io_map, oo_map, nv_map);

  auto evaluated_info = evaluate_with_missions(model,
                                               node_io_map,
                                               oo_map,
                                               nv_map);
  auto output_form = evaluated_info.first;

  /**
   * @brief hw 2-4-2: Calculate memory requirements for storing the activations
   */
  {
    int64_t total_activation_bytes = 0;
    for (size_t i = 0; i < output_form.size(); ++i) {
      int sz = output_form[i].first;
      for (auto &d: output_form[i].second)
        sz *= d;
      total_activation_bytes += sz;
    }
    
    std::vector<std::string> output_sizes;
    { /* Main answer */
      std::stringstream ss;
      ss << "Total memory for activations: " << total_activation_bytes << " bytes";
      output_sizes.push_back(ss.str());
      output_sizes.push_back("Output size of each layers...[ SHAPE ] (ELEMENT_SIZE)");
    }

    /* Additional information: output shape */
    for (size_t i = 0; i < output_form.size(); ++i) {
      std::stringstream ss;
      ss << "\t[";
      auto lim = output_form[i].second.size();
      for (size_t j = 0; j < lim; ++j)
        ss << output_form[i].second.vec()[j] << (j + 1 == lim ? "" : ", ");
      ss << "] (" << output_form[i].first << ")";
      output_sizes.push_back(ss.str());
    }
    
    print_hw_result("2-4-2",
                    "Calculate memory requirements for storing the activations",
                    output_sizes);
  }

  /**
   * @brief hw 2-4-3: Calculate computation requirements
   */
  auto res = evaluated_info.second;

  std::vector<std::string> info;
  {
    std::stringstream ss;
    ss << "Conv: " << res.first << " macs";
    info.push_back(ss.str());
  }
  {
    std::stringstream ss;
    ss << "Linear: " << res.second << " macs";
    info.push_back(ss.str());
  }

  print_hw_result("2-4-3",
                  "Calculate computation requirements",
                  info);
}