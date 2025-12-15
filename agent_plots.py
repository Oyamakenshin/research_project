import matplotlib.pyplot as plt
import seaborn as sns

def visualize_agent_parameters(model):
    """
    エージェントの初期パラメータ分布を可視化する。（詳細版）
    αとλの分布も表示する。

    Args:
        model: 初期化済みのMesaモデルインスタンス
    """
    agent_df = model.datacollector.get_agent_vars_dataframe()

    if 0 not in agent_df.index.get_level_values('Step'):
        print("ステップ0のデータが見つかりません。モデルが正しく初期化されているか確認してください。")
        return
        
    initial_params = agent_df.loc[0]

    try:
        sns.set_theme(style="whitegrid", font='Hiragino Sans')
    except RuntimeError:
        print("日本語フォント'Hiragino Sans'が見つかりません。英語表示にフォールバックします。")
        sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('エージェントの初期パラメータ分布 (詳細)', fontsize=20)

    # 1. Wealth（資産）の分布
    sns.histplot(data=initial_params, x='Wealth', ax=axes[0, 0], kde=True)
    axes[0, 0].set_title('Wealth (資産) の分布', fontsize=14)
    axes[0, 0].set_xscale('log')

    # 2. w_1 (独占時の効用) の分布
    sns.histplot(data=initial_params, x='w_1', ax=axes[0, 1], kde=True)
    axes[0, 1].set_title('w_1 (独占時の効用) の分布', fontsize=14)
    axes[0, 1].set_xscale('log')

    # 3. Persona（ペルソナ）の分布
    sns.countplot(data=initial_params, x='Persona', ax=axes[1, 0], palette='viridis', order=['Bandwagon', 'Neutral', 'Snob'])
    axes[1, 0].set_title('Persona (ペルソナ) の分布', fontsize=14)

    # 4. Alpha (α) の分布（ペルソナ別）
    sns.histplot(data=initial_params, x='alpha', hue='Persona', ax=axes[1, 1], kde=True, multiple="stack", palette='viridis', hue_order=['Bandwagon', 'Neutral', 'Snob'])
    axes[1, 1].set_title('Alpha (α) の分布（ペルソナ別）', fontsize=14)

    # 5. Lambda (λ) の分布（ペルソナ別）
    sns.histplot(data=initial_params, x='lamb', hue='Persona', ax=axes[2, 0], kde=True, multiple="stack", palette='viridis', hue_order=['Bandwagon', 'Neutral', 'Snob'])
    axes[2, 0].set_title('Lambda (λ) の分布（ペルソナ別）', fontsize=14)

    # 6. Wealth vs w_1 の散布図
    sns.scatterplot(data=initial_params, x='Wealth', y='w_1', hue='Persona', ax=axes[2, 1], alpha=0.7, palette='viridis', hue_order=['Bandwagon', 'Neutral', 'Snob'])
    axes[2, 1].set_title('Wealth vs. w_1', fontsize=14)
    axes[2, 1].set_xscale('log')
    axes[2, 1].set_yscale('log')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


def visualize_initial_distributions(model):
    """
    シミュレーションの初期状態(Step 0)における主要なエージェントパラメータの分布を可視化する。
    W1, Wealth, Ln-1, Wn に特化。

    Args:
        model: 初期化済みのMesaモデルインスタンス
    """
    agent_df = model.datacollector.get_agent_vars_dataframe()

    if 0 not in agent_df.index.get_level_values('Step'):
        print("ステップ0のデータが見つかりません。モデルが正しく初期化されているか確認してください。")
        return
        
    initial_params = agent_df.loc[0]

    try:
        sns.set_theme(style="whitegrid", font='Hiragino Sans')
    except RuntimeError:
        print("日本語フォント'Hiragino Sans'が見つかりません。英語表示にフォールバックします。")
        sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('エージェント初期パラメータの分布', fontsize=20)

    # 1. W1 (独占価値) の分布
    sns.histplot(data=initial_params, x='w_1', ax=axes[0, 0], kde=True, color='skyblue')
    axes[0, 0].set_title('W1 (独占価値) の分布', fontsize=14)
    axes[0, 0].set_xlabel('W1 (独占価値)')
    axes[0, 0].set_xscale('log')

    # 2. Wealth (資産) の分布
    sns.histplot(data=initial_params, x='Wealth', ax=axes[0, 1], kde=True, color='lightgreen')
    axes[0, 1].set_title('Wealth (資産) の分布', fontsize=14)
    axes[0, 1].set_xlabel('Wealth (資産)')
    axes[0, 1].set_xscale('log')

    # 3. Ln-1 (最大疎外感) の分布（ペルソナ別）
    sns.boxplot(data=initial_params, x='Persona', y='l_n_minus_1', ax=axes[1, 0], palette='viridis', order=['Bandwagon', 'Neutral', 'Snob'])
    axes[1, 0].set_title('Ln-1 (最大疎外感) の分布（ペルソナ別）', fontsize=14)
    axes[1, 0].set_xlabel('ペルソナ')
    axes[1, 0].set_ylabel('Ln-1 (最大疎外感)')

    # 4. Wn (飽和価値) の分布（ペルソナ別）
    sns.violinplot(data=initial_params, x='Persona', y='w_n', ax=axes[1, 1], palette='viridis', order=['Bandwagon', 'Neutral', 'Snob'])
    axes[1, 1].set_title('Wn (飽和価値) の分布（ペルソナ別）', fontsize=14)
    axes[1, 1].set_xlabel('ペルソナ')
    axes[1, 1].set_ylabel('Wn (飽和価値)')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


import pandas as pd # Ensure pandas is imported if not already

def visualize_w1_wealth_scatter(results_df):
    """
    シミュレーション結果から、W1 (独占価値) と Wealth (資産) の散布図を可視化する。
    ペルソナ別に色分けし、対数スケールで表示する。

    Args:
        results_df (pd.DataFrame): batch_runの出力結果DataFrame。
                                   'w_1', 'Wealth', 'Persona'カラムを含むこと。
    """
    # 各エージェントの初期パラメータはRunIdとAgentIDで一意なので、重複を除去
    # Step 0のデータを使用するのが最も適切
    initial_agent_params = results_df[results_df['Step'] == 0].drop_duplicates(subset=['RunId', 'AgentID'])

    if initial_agent_params.empty:
        print("可視化するデータが見つかりません。results_dfにStep 0のデータが含まれているか確認してください。")
        return

    try:
        sns.set_theme(style="whitegrid", font='Hiragino Sans')
    except RuntimeError:
        print("日本語フォント'Hiragino Sans'が見つかりません。英語表示にフォールバックします。")
        sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=initial_agent_params,
        x='Wealth',
        y='w_1',
        hue='Persona',
        palette='viridis',
        alpha=0.7,
        s=50, # Adjust point size
        hue_order=['Bandwagon', 'Neutral', 'Snob']
    )

    plt.title('W1 (独占価値) vs Wealth (資産) の散布図', fontsize=16)
    plt.xlabel('Wealth (資産) (Log Scale)', fontsize=12)
    plt.ylabel('W1 (独占価値) (Log Scale)', fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.legend(title='Persona')
    plt.tight_layout()
    plt.show()