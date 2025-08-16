import subprocess
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

def run_ncu_benchmark(m, n, k):
    try:
        result = subprocess.run(
            [
                'ncu', '--csv', '--cache-control=all', '--clock-control=base',
                '--metrics', 'gpu__time_duration.sum',
                '--', './hgemm', f'-M={m}', f'-N={n}', f'-K={k}'
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        output = result.stdout

        if "==Error==" in output or "Error" in output:
            raise RuntimeError("Error in ncu process")

        csv_start = output.find('"ID","Process ID"')
        if csv_start == -1:
            raise ValueError("CSV data not found in ncu output")

        csv_data = output[csv_start:]
        df = pd.read_csv(StringIO(csv_data))[['Kernel Name', 'Metric Value']]
        df['M'] = m
        df['Time (ms)'] = (df['Metric Value'] / 1_000_000).round(2)
        df.drop(columns=['Metric Value'], inplace=True)
        df['TFLOPS'] = ((2 * m * n * k) / (df['Time (ms)'] * 1e9)).round(2)
        return df

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ncu command failed with error: {e.stderr}")
    
def plot_benchmark_results(dataframes, output_file="benchmark_results.png"):
    combined_df = pd.concat(dataframes).reset_index(drop=True)
    
    # Find the top 5 kernels that are present in the most unique M values
    kernel_presence = combined_df.groupby('Kernel Name')['M'].nunique()
    top_kernels = kernel_presence.nlargest(5).index.tolist()
    
    # Ensure kernels starting with 'void hgemm' are always included
    hgemm_kernels = combined_df[combined_df['Kernel Name'].str.startswith('void hgemm')]['Kernel Name'].unique()
    selected_kernels = list(set(top_kernels) | set(hgemm_kernels))
    
    filtered_df = combined_df[combined_df['Kernel Name'].isin(selected_kernels)]
    
    grouped = filtered_df.groupby(['M', 'Kernel Name'])['TFLOPS'].mean().reset_index()
    
    plt.figure(figsize=(12, 8))
    for kernel_name in grouped['Kernel Name'].unique():
        kernel_data = grouped[grouped['Kernel Name'] == kernel_name]
        if kernel_name.startswith('void hgemm'):
            # Bolden the line for 'void hgemm' kernels
            plt.plot(kernel_data['M'], kernel_data['TFLOPS'], label=kernel_name, linewidth=3.0)
        else:
            plt.plot(kernel_data['M'], kernel_data['TFLOPS'], label=kernel_name, linewidth=1.5)

    plt.xlabel("Dimension")
    plt.ylabel("TFLOPS")
    plt.grid(True)
    
    # Split the legend into 2 columns
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize='small', frameon=False)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')  # Ensure the legend is included in the saved image
    print(f"Plot saved to {output_file}")
    plt.close()

def main():
  # Define parameter ranges
  M_values = [i for i in range(8192, 20 * 8192, 2048)]
  N_values = [256 for i in range(8192, 20 * 8192, 2048)]
  K_values = [256 for i in range(8192, 20 * 8192, 2048)]

  #M_values = [i for i in range(256, 12288, 256)]
  #N_values = M_values
  #K_values = M_values
  
  results = []

  for m, n, k in zip(M_values, N_values, K_values):
    print(f"Running M={m}, N={n}, K={k}...")
    df = run_ncu_benchmark(m, n, k)
    print(df)
    print('-'*10)
    results.append(df)

  plot_benchmark_results(results)

if __name__ == "__main__":
    main()