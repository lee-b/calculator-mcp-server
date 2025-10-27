# Mathematical Calculator MCP Server

This is a Model Context Protocol (MCP) server that provides Claude with advanced mathematical calculation capabilities, including symbolic math, statistical analysis, and matrix operations.

This is a (heavily modified, not rewritten) version of https://github.com/huhabla/calculator-mcp-server. Mostly I had AI code the addition of heavy testing and bug fixes and until it worked.

## Features

The Mathematical Calculator MCP Server provides the following tools:

- **Basic Calculations**: Evaluate mathematical expressions safely
- **Symbolic Mathematics**:
  - Solve equations (linear, quadratic, polynomial, etc.)
  - Calculate derivatives of expressions
  - Compute integrals of expressions
- **Statistical Analysis**:
  - Mean, median, mode
  - Variance, standard deviation
  - Correlation coefficient
  - Linear regression
  - Confidence intervals
- **Matrix Operations**:
  - Matrix addition
  - Matrix multiplication
  - Matrix transposition

## Installation

### Prerequisites

- Python 3.10+ (recommended: Python 3.11+)
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Internet access for uv to download other dependencies during install

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/lee-b/calculator-mcp-server.git
   cd calculator-mcp-server
   ```

2. (Option 1) Setup with the provided script:
   ```bash
   chmod +x setup_venv.sh
   ./setup_venv.sh
   ```

   (Option 2) Or manually set up the virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Run doc-tests to verify everything works:
   ```bash
   bash run_doctests.sh
   ```

## Integration with Claude Desktop

### Method 1: Configure in stdio-capable MCP client

Add the server to your MCP client configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`  
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Add the following to the `mcpServers` section:

```json
{
  "mcpServers": {
    "calculator": {
      "command": "uvx",
      "args": [
        "--from",
        "calculator-mcp-server@git+https://github.com/lee-b/calculator-mcp-server.git",
        "--",
        "calculator-mcp-server",
        "--stdio"
      ]
    }
  }
}
```

**Note**: The `--stdio` flag is required for proper integration with Claude Desktop. The `--` separates uvx arguments from the calculator server arguments.

### Method 2: Install with FastMCP

1. Make sure you have uv installed ([Installation Guide](https://github.com/astral-sh/uv))

2. Install the MCP server in Claude Desktop:
   ```bash
   fastmcp install calculator_server.py
   ```

   Or with a custom name:
   ```bash
   fastmcp install calculator_server.py --name "Math Calculator"
   ```

3. Once installed, your AI should automatically have access to all the mathematical tools and functions.

## Usage Examples

After integrating with your AI, you should be able to ask IT to perform various mathematical operations. Here are some examples:

### Basic Calculations
```
Can you calculate 3.5^2 * sin(pi/4)?
```

### Solving Equations
```
Solve the following equation: x^2 - 5x + 6 = 0
```

### Calculating Derivatives
```
What's the derivative of sin(x^2) with respect to x?
```

### Computing Integrals
```
Calculate the integral of x^2 * e^x
```

### Statistical Analysis
```
Find the mean, median, mode, and standard deviation of this dataset: [23, 45, 12, 67, 34, 23, 18, 95, 41, 23]
```

### Linear Regression
```
Perform a linear regression on these points: (1,2), (2,3.5), (3,5.1), (4,6.5), (5,8.2)
```

### Matrix Operations
```
Multiply these two matrices:
[1, 2, 3]
[4, 5, 6]

and

[7, 8]
[9, 10]
[11, 12]
```

## Advanced Usage Examples

### Complex Expressions

- Calculate the indefinite integral: "x^2 * sin(x)"
- Solve an equation: "x^2 - 5x + 6 = 0"
- Differentiate a function: "ln(x^2 + 1)"

Note: The server requires mathematical expressions as input, not free natural language.

### Matrix and Vector Operations

- Matrix inversion: "Invert the matrix [[1,2],[3,4]]"
- Vector normalization: "Normalize the vector [3,4,5]"
- Matrix eigenvalue: "Find eigenvalues of [[2,1],[1,2]]"

### Statistical Computations

- Hypothesis testing: "Perform t-test on datasets [1,2,3] and [4,5,6]"
- ANOVA: "Compute ANOVA for groups [[1,2],[3,4],[5,6]]"
- Regression analysis: "Fit polynomial regression of degree 2 to points [(1,1),(2,4),(3,9)]"

## Limitations and Notes

- **Safe Evaluation Restrictions**: Expressions are evaluated in a restricted environment with only whitelisted mathematical functions and constants to prevent security vulnerabilities. Arbitrary code execution is not allowed.
- **Plotting Display Requirements**: Plotting functions require a graphical display environment (e.g., X11 on Linux, or a compatible setup). Plots may not display in headless environments.
- **Input Data Types**: All numerical inputs must be provided as floats or integers. Lists and tuples are accepted for datasets, matrices, and vectors. Invalid data types will result in errors.

## Tool Quick Reference

| Category              | Tools                                                                 |
|-----------------------|-----------------------------------------------------------------------|
| Basic Calculations    | calculate                                                            |
| Symbolic Mathematics  | solve_equation, differentiate, integrate, expand, factorize          |
| Statistical Analysis  | mean, variance, standard_deviation, median, mode, correlation_coefficient, linear_regression, confidence_interval |
| Matrix Operations     | matrix_addition, matrix_multiplication, matrix_transpose, matrix_determinant |
| Vector Operations     | vector_dot_product, vector_cross_product, vector_magnitude           |
| Plotting              | plot_function                                                        |
| Other                 | summation                                                            |

## Development

### Testing

Run `uv run pytest`


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [FastMCP](https://github.com/jlowin/fastmcp) for the Pythonic MCP server framework
- [SymPy](https://sympy.org/) for symbolic mathematics
- [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/) for numerical and statistical computations
- https://github.com/huhabla/calculator-mcp-server - the original from which this was forked (and heavily modified, but not rewritten)
