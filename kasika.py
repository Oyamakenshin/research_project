import matplotlib.pyplot as plt
import seaborn as sns

def visualize_agent_parameters(model):
    """
    エージェントの初期パラメータ分布を可視化する。

    Args:
        model: 初期化済みのMesaモデルインスタンス
    """
    # DataCollectorからエージェントのデータをDataFrameとして取得
    agent_df = model.datacollector.get_agent_vars_dataframe()

    # Step 0のデータ（初期状態）のみを抽出
    if 0 not in agent_df.index.get_level_values('Step'):
        print("ステップ0のデータが見つかりません。モデルが正しく初期化されているか確認してください。")
        return
        
    initial_agent_params = agent_df.loc[0]

    # --- 可視化 ---
    # 日本語表示のためのフォント設定
    # ご自身の環境に合わせてフォント名を変更してください (例: 'Meiryo', 'MS Gothic'など)
    try:
        sns.set_theme(style="whitegrid", font='Hiragino Sans')
    except RuntimeError:
        print("日本語フォント'Hiragino Sans'が見つかりません。'Meiryo'や'MS Gothic'などに変更するか、英語表示にフォールバックします。")
        sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('エージェントの初期パラメータ分布 (Initial Agent Parameter Distributions)', fontsize=20)

    # 1. Wealth（資産）の分布
    sns.histplot(data=initial_agent_params, x='Wealth', ax=axes[0, 0], kde=True)
    axes[0, 0].set_title('Wealth (資産) の分布', fontsize=14)
    axes[0, 0].set_xscale('log') # パレート分布は対数スケールで確認

    # 2. w_1 (独占時の効用) の分布
    sns.histplot(data=initial_agent_params, x='w_1', ax=axes[0, 1], kde=True)
    axes[0, 1].set_title('w_1 (独占時の効用) の分布', fontsize=14)
    axes[0, 1].set_xscale('log') # パレート分布は対数スケールで確認

    # 3. Persona（ペルソナ）の分布
    sns.countplot(data=initial_agent_params, x='Persona', ax=axes[1, 0], palette='viridis', order=['Bandwagon', 'Neutral', 'Snob'])
    axes[1, 0].set_title('Persona (ペルソナ) の分布', fontsize=14)

    # 4. Alpha (α) の分布（ペルソナ別）
    sns.histplot(data=initial_agent_params, x='alpha', hue='Persona', ax=axes[1, 1], kde=True, multiple="stack", palette='viridis', hue_order=['Bandwagon', 'Neutral', 'Snob'])
    axes[1, 1].set_title('Alpha (α) の分布（ペルソナ別）', fontsize=14)

    # 5. Lambda (λ) の分布（ペルソナ別）
    sns.histplot(data=initial_agent_params, x='lamb', hue='Persona', ax=axes[2, 0], kde=True, multiple="stack", palette='viridis', hue_order=['Bandwagon', 'Neutral', 'Snob'])
    axes[2, 0].set_title('Lambda (λ) の分布（ペルソナ別）', fontsize=14)

    # 6. Wealth vs w_1 の散布図
    sns.scatterplot(data=initial_agent_params, x='Wealth', y='w_1', hue='Persona', ax=axes[2, 1], alpha=0.7, palette='viridis', hue_order=['Bandwagon', 'Neutral', 'Snob'])
    axes[2, 1].set_title('Wealth vs. w_1', fontsize=14)
    axes[2, 1].set_xscale('log')
    axes[2, 1].set_yscale('log')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
