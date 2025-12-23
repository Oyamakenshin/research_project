from mesa import Model, Agent
from mesa.datacollection import DataCollector
import numpy as np

# ===========================================================
# 参加者エージェント（データ購入者）
# ===========================================================
class Participants(Agent):
    def __init__(self, model):
        super().__init__(model)
        # 各エージェントの状態変数
        self.has_token = False      # トークン（データ）をすでに購入しているか
        self.bought_step = None     # 購入したステップ番号（タイムステップ）
        self.bid_flag = False       # 購入希望フラグ
        self.persona = None         # 性格タイプ ("intrinsic" or "follower")
        self.wealth = 0             # 現在の資産
        self.primary_wealth = 0     # 初期資産（基準値保持用）
        self.w_1 = 0                # 独占状態での効用
        self.alpha = 0              
        self.w_n = 0                # 市場全員が保有したときの効用
        self.lamb = 0
        self.l_n_minus_1 = 0        # 他者が保有しているときの効用損失（負の値）
        self.current_utility = 0      # 現在の効用値
        self.bought_this_step = False  # 今ステップで購入したかどうかのフラグ

    # -------------------------------------------------------
    # エージェントの初期パラメータを設定（個人属性生成）
    # -------------------------------------------------------
    def initialize_attributes(self, persona):
        self.persona = persona
        
        # 資産（Pareto分布に従って生成）: αが大きいほど均等、scaleはスケール
        self.wealth = self.random.paretovariate(self.model.wealth_alpha) * self.model.wealth_scale
        self.primary_wealth = self.wealth

        # w1（独占効用）もPareto分布から生成
        self.w_1 = self.random.paretovariate(self.model.w1_alpha) * self.model.w1_scale

        # ペルソナ（性格タイプ）によって、効用構造を変える
        if self.persona == "Bandwagon":
            # Bandwagon型: みんなが持つほど欲しくなる
            self.lamb = self.random.uniform(0.7, 0.9)
            self.l_n_minus_1 = -self.w_1 * self.lamb  # 他者保有で効用が大きく下がる
            self.alpha = self.random.uniform(0.70, 0.90)        # w_nはw_1の70〜90%程度
            self.w_n = self.w_1 * self.alpha
        
        elif self.persona == "Neutral":
            # Neutral: みんなが持っていてもあまり気にしない
            self.lamb = self.random.uniform(0.4, 0.6)
            self.l_n_minus_1 = -self.w_1 * self.lamb
            self.alpha = self.random.uniform(0.4, 0.6)   
            self.w_n = self.w_1 * self.alpha
            
        elif self.persona == 'Snob':
            # Snob: みんなが持っていると欲しくなくなる
            self.lamb = self.random.uniform(0.1, 0.3)
            self.l_n_minus_1 = -self.w_1 * self.lamb
            self.alpha = self.random.uniform(0.1, 0.3)        
            self.w_n = self.w_1 * self.alpha
            
        # 負の値や不正値を防ぐ
        self.wealth = max(0, self.wealth)
        self.w_1 = max(0, self.w_1)
        self.w_n = max(0, self.w_n)
        if self.l_n_minus_1 > 0:
            self.l_n_minus_1 = -self.l_n_minus_1  # 強制的に負にして「損失項」として扱う
        
        #print(f"Agent {self.unique_id} initialized: Persona={self.persona}, Wealth={self.wealth:.2f}, w_1={self.w_1:.2f}, w_n={self.w_n:.2f}, l_n-1={self.l_n_minus_1:.2f}")

    # -------------------------------------------------------
    # 現在の販売数kに応じた「期待効用」から入札価格（bid）を計算
    # -------------------------------------------------------
    def calculate_bid(self, k, n):
        """
        Calculate the bid amount based on non-linear utility (W) and loss (L) functions.

        定義:
            W(k) = W1 * r^(k-1), where r = alpha^(1/(n-1))
            L(k) = -lambda * (1/(1 - alpha)) * (W1 - W(k+1))
            bid(k) = W(k+1) - L(k)

        Parameters
        ----------
        k : int
            Number of items already sold (0 <= k <= n-1)
        n : int
            Total number of items
        """
        # パラメータの取得
        alpha = self.alpha
        lambda_ = self.lamb
        W1 = self.w_1
        # 幾何減衰率 r
        r = alpha ** (1 / (n - 1))
        # 効用 W(k+1)
        W_k_plus_1 = W1 * (r ** k)
        # 損失 L(k)
        L_k = -lambda_ * (1 / (1 - alpha)) * (W1 - W1 * (r ** k))
        # 入札額（効用差）
        bid = W_k_plus_1 - L_k
        #print(f"Agent {self.unique_id} bid calculation at k={k}: W(k+1)={W_k_plus_1:.2f}, L(k)={L_k:.2f}, bid={bid:.2f}")
        return bid
    
    # -------------------------------------------------------
    # 買いたいと思うかどうかを確率的に判定
    # -------------------------------------------------------
    def flag_if_interested(self, k):
        price = self.model.current_price
        bid_price = self.calculate_bid(k, self.model.num_agents)

        # トークンが売り切れ or すでに持っている場合はスキップ
        if k >= self.model.num_data or self.has_token:
            return False

        # 購入確率を「bid_price / price」比率からロジスティック関数で決定
        r = bid_price / (price + 1e-9)               # 安全に割り算
        x = np.log(r) / max(self.model.tau, 1e-9)    # tauが感度パラメータ（小さいほど急峻）
        prob_buy = 1 / (1 + np.exp(np.clip(-x, -60, 60)))  # ロジスティック関数
        
        # ランダムに購入するかどうかを決定
        # if self.random.random() < prob_buy and price <= self.wealth:
        if self.random.random() < prob_buy:
            self.bid_flag = True
            return True
        return False
        
    # -------------------------------------------------------
    # 実際に購入を実行
    # -------------------------------------------------------
    def buy_if_flagged(self):
        if self.bid_flag and not self.has_token and self.model.sold_tokens < self.model.num_data:
            # 支払い処理
            self.wealth -= self.model.current_price
            self.has_token = True
            # モデル全体の販売数・収益を更新
            self.model.sold_tokens += 1
            self.model.provider_revenue += self.model.current_price
            # 状態更新
            self.bid_flag = False
            self.bought_step = self.model.steps
            self.bought_this_step = True
            return True
        else:
            self.bid_flag = False
            return False
    def calculate_current_utility(self, k_before, k_after):
        """
        現在の販売数kに基づいて、エージェントの現在の効用を計算する。
        """
        n = self.model.num_agents
        alpha = self.alpha
        lambda_ = self.lamb
        W1 = self.w_1
        r = alpha ** (1 / (n - 1))
        if self.has_token:
            current_utility = W1 * (r ** (k_after - 1)) + lambda_ * (1 / (1 - alpha)) * (W1 - W1 * (r ** k_before)) - self.model.current_price
        else:
            current_utility = -lambda_ * (1 / (1 - alpha)) * (W1 - W1 * (r ** k_after))
            
        # if current_utility != 0:
        #     print(f"Agent {self.unique_id} current utility at k={k}, r={r}, W1={W1}, lambda={lambda_}, initial_price={self.model.initial_price}: {current_utility}")
        self.current_utility = current_utility
        self.bought_this_step = False  # リセット
        return

# ===========================================================
# モデルクラス：DataMarket
# ===========================================================
class DataMarket(Model):
    """
    一次市場（Data Provider vs Participants）をシミュレートするABMモデル。
    各エージェントが効用に基づき確率的にデータを購入する。
    """

    def __init__(self, num_agents, num_data, initial_price, 
                 persona_dist, wealth_alpha, wealth_scale, w1_params, 
                 tau=0.5, seed=None, dynamic_pricing=False, gamma=0.0):
        super().__init__(seed=seed)

        # モデル全体の基本設定
        self.num_agents = num_agents      # 総エージェント数
        self.num_data = num_data          # トークン供給量（データ販売数）
        self.initial_price = initial_price
        self.persona_dist = persona_dist  # {"intrinsic":0.5, "follower":0.5} のような比率
        self.tau = tau                    # ロジスティック関数の鋭さ
        self.dynamic_pricing = dynamic_pricing
        self.gamma = gamma
        self.current_price = initial_price # 現在の価格（動的に変動）

        # 分布パラメータ
        self.wealth_alpha, self.wealth_scale = wealth_alpha, wealth_scale
        self.w1_alpha, self.w1_scale = w1_params # w1パラメータをタプルで受け取る
        
        # 初期状態
        self.steps = 0
        self.provider_revenue = 0         # プロバイダ（販売者）の収益
        self.sold_tokens = 0              # 販売済みトークン数
        
        # ===================================================
        # データ収集設定
        # ===================================================
        self.datacollector = DataCollector(
            model_reporters={
                "Holders": lambda m: m.sold_tokens,        # 保有者数（販売済みトークン）
                "ProviderRevenue": "provider_revenue",      # プロバイダ収益
                "CurrentPrice": "current_price"             # 現在価格
            },
            agent_reporters={
                "w_1": "w_1",
                "w_n": "w_n",
                "alpha": "alpha",
                "lamb": "lamb",
                "l_n_minus_1": "l_n_minus_1",
                "HasToken": "has_token",
                "Wealth": "wealth",
                "Persona": "persona",
                "BoughtStep": "bought_step",
                "CurrentUtility": "current_utility"
            }
        )
        
        # ===================================================
        # エージェントの生成と初期化
        # ===================================================
        self.participants = Participants.create_agents(self, n=num_agents)

        # personaの比率に応じて分類
        num_Bandwagon = int(self.num_agents * self.persona_dist.get('Bandwagon', 0))
        self.num_intrinsic = num_Bandwagon
        num_Neutral = int(self.num_agents * self.persona_dist.get('Neutral', 0))
        self.num_follower = num_Neutral
        num_Snob = self.num_agents - num_Bandwagon - num_Neutral
        self.num_snob = num_Snob
                
        # エージェントをシャッフルしてランダムに割り当て
        agent_list = list(self.participants)
        self.random.shuffle(agent_list)

        # intrinsic → follower の順で割り当て
        for i, agent in enumerate(agent_list):
            if i < num_Bandwagon:
                persona = "Bandwagon"
            elif i < num_Bandwagon + num_Neutral:
                persona = "Neutral"
            else:
                persona = "Snob"
            agent.initialize_attributes(persona)
        
        # 初期データ収集
        self.datacollector.collect(self)

    def update_price(self):
        """
        動的価格設定が有効な場合、販売数に応じて価格を更新する。
        P(k) = P_base * (1 + gamma * k / N)
        """
        if self.dynamic_pricing:
            self.current_price = self.initial_price * (1 + self.gamma * (self.sold_tokens / self.num_agents))

    # -------------------------------------------------------
    # 1ステップ分のシミュレーション（逐次意思決定モデル）
    # -------------------------------------------------------
    def step(self):
        k_before = self.sold_tokens  # Step開始時の販売数
        
        # まだ購入していないエージェントを取得
        non_holder_agents = self.participants.select(lambda agent: not agent.has_token)
        
        # ランダムな順序で意思決定させる（逐次処理）
        # shuffle_doはリスト全体に対する並行処理的な意味合いが強いため、
        # ここでは明示的にリストを取得してシャッフルし、forループで回す
        agent_list = list(non_holder_agents)
        self.random.shuffle(agent_list)
        
        for agent in agent_list:
            # 1. 現在の販売数(k)に基づいて興味判定
            # 注意: ここでの k は「この瞬間の」sold_tokens を使うべき
            current_k = self.sold_tokens
            
            interested = agent.flag_if_interested(k=current_k)
            
            if interested:
                # 2. 購入試行
                bought = agent.buy_if_flagged()
                
                # 3. 購入が発生したら即座に価格更新 (Sequential Update)
                if bought:
                    current_k += 1
                    self.update_price()
                    agent.calculate_current_utility(k_before=k_before, k_after=current_k)
        
        # --- 統計情報の更新（効用計算など） ---
        # 逐次処理が終わった後の状態(k_after)で、対象エージェントの効用を再計算・記録する
        # 対象: 「このステップで購入した人」 または 「まだ持っていない人」
        # （既に持っている人は、購入時点の効用で固定するという仮定のため更新しない）
        
        k_after = self.sold_tokens
        
        # 対象エージェントを抽出
        # Note: bought_this_step is True ONLY for agents who bought in this specific step
        target_agents = self.participants.select(
            lambda agent: not agent.has_token
        )
        
        # 対象者のみ効用計算を実行
        target_agents.shuffle_do("calculate_current_utility", k_before=k_before, k_after=k_after)
        
        # 結果をデータ収集
        self.datacollector.collect(self)
