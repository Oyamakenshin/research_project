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
        self.w_n = 0                # 市場全員が保有したときの効用
        self.l_n_minus_1 = 0        # 他者が保有しているときの効用損失（負の値）

    # -------------------------------------------------------
    # エージェントの初期パラメータを設定（個人属性生成）
    # -------------------------------------------------------
    def initialize_attributes(self, persona):
        self.persona = persona
        
        # 資産（Pareto分布に従って生成）: αが大きいほど均等、scaleはスケール
        self.wealth = self.model.base_wealth + (self.random.paretovariate(self.model.wealth_alpha) - 1) * self.model.wealth_scale
        self.primary_wealth = self.wealth

        # w1（独占効用）もPareto分布から生成
        self.w_1 = (self.random.paretovariate(self.model.w1_alpha) - 1) * self.model.w1_scale

        # ペルソナ（性格タイプ）によって、効用構造を変える
        if self.persona == "Bandwagon":
            # Bandwagon型: みんなが持つほど欲しくなる
            l_multiplier = self.random.uniform(0.1, 0.3)
            self.l_n_minus_1 = -self.w_1 * l_multiplier  # 他者保有で効用が大きく下がる
            wn_multiplier = self.random.uniform(0.70, 0.90)        # w_nはw_1の70〜90%程度
            self.w_n = self.w_1 * wn_multiplier
        
        elif self.persona == "Neutral":
            # Neutral: みんなが持っていてもあまり気にしない
            l_multiplier = self.random.uniform(0.4, 0.6)
            self.l_n_minus_1 = -self.w_1 * l_multiplier 
            wn_multiplier = self.random.uniform(0.4, 0.6)   
            
        elif self.persona == 'Snob':
            # Snob: みんなが持っていると欲しくなくなる
            l_multiplier = self.random.uniform(0.3, 0.5)
            self.l_n_minus_1 = -self.w_1 * l_multiplier 
            wn_multiplier = self.random.uniform(0.1, 0.3)        
            self.w_n = self.w_1 * wn_multiplier
            
        # 負の値や不正値を防ぐ
        self.wealth = max(0, self.wealth)
        self.w_1 = max(0, self.w_1)
        self.w_n = max(0, self.w_n)
        if self.l_n_minus_1 > 0:
            self.l_n_minus_1 = -self.l_n_minus_1  # 強制的に負にして「損失項」として扱う
        
        print(f"Agent {self.unique_id} initialized: Persona={self.persona}, Wealth={self.wealth:.2f}, w_1={self.w_1:.2f}, w_n={self.w_n:.2f}, l_n-1={self.l_n_minus_1:.2f}")

    # -------------------------------------------------------
    # 現在の販売数kに応じた「期待効用」から入札価格（bid）を計算
    # -------------------------------------------------------
    def calculate_bid(self, k, n):
        # k個がすでに販売済み → 自分が(n−k)番目の購入者と仮定して効用を補間
        # w_k+1: k個販売時点での効用（線形補間）
        w_k_plus_1 = self.w_1 - (self.w_1 - self.w_n) * (k / (n - 1))
        # l_k: 他者がk個保有している時点での損失（負の値）
        l_k = self.l_n_minus_1 * (k / (n - 1))
        # 実際の入札額（=効用差）
        return w_k_plus_1 - l_k

    # -------------------------------------------------------
    # 買いたいと思うかどうかを確率的に判定
    # -------------------------------------------------------
    def flag_if_interested(self, k):
        price = self.model.initial_price
        bid_price = self.calculate_bid(k, self.model.num_agents)

        # トークンが売り切れ or すでに持っている場合はスキップ
        if k >= self.model.num_data or self.has_token:
            return False

        # 購入確率を「bid_price / price」比率からロジスティック関数で決定
        r = bid_price / (price + 1e-9)               # 安全に割り算
        x = np.log(r) / max(self.model.tau, 1e-9)    # tauが感度パラメータ（小さいほど急峻）
        prob_buy = 1 / (1 + np.exp(np.clip(-x, -60, 60)))  # ロジスティック関数
        
        # ランダムに購入するかどうかを決定
        if self.random.random() < prob_buy and price <= self.wealth:
            self.bid_flag = True
            return True
        return False
        
    # -------------------------------------------------------
    # 実際に購入を実行
    # -------------------------------------------------------
    def buy_if_flagged(self):
        if self.bid_flag and not self.has_token and self.model.sold_tokens < self.model.num_data:
            # 支払い処理
            self.wealth -= self.model.initial_price
            self.has_token = True
            # モデル全体の販売数・収益を更新
            self.model.sold_tokens += 1
            self.model.provider_revenue += self.model.initial_price
            # 状態更新
            self.bid_flag = False
            self.bought_step = self.model.steps
            return True
        else:
            self.bid_flag = False
            return False

# ===========================================================
# モデルクラス：DataMarket
# ===========================================================
class DataMarket(Model):
    """
    一次市場（Data Provider vs Participants）をシミュレートするABMモデル。
    各エージェントが効用に基づき確率的にデータを購入する。
    """

    def __init__(self, num_agents, num_data, initial_price, base_wealth, 
                 persona_dist, wealth_alpha, wealth_scale, w1_params, 
                 tau=0.5, seed=None):
        super().__init__(seed=seed)

        # モデル全体の基本設定
        self.num_agents = num_agents      # 総エージェント数
        self.num_data = num_data          # トークン供給量（データ販売数）
        self.initial_price = initial_price
        self.base_wealth = base_wealth
        self.persona_dist = persona_dist  # {"intrinsic":0.5, "follower":0.5} のような比率
        self.tau = tau                    # ロジスティック関数の鋭さ

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
                "ProviderRevenue": "provider_revenue"      # プロバイダ収益
            },
            agent_reporters={
                "w_1": lambda a: a.w_1,
                "w_n": lambda a: a.w_n,
                "l_n_minus_1": lambda a: a.l_n_minus_1,
                "HasToken": lambda a: a.has_token,
                "Wealth": lambda a: a.wealth,
                "Persona": lambda a: a.persona,
                "BoughtStep": lambda a: a.bought_step
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

    # -------------------------------------------------------
    # 1ステップ分のシミュレーション
    # -------------------------------------------------------
    def step(self):
        k = self.sold_tokens  # 現在の販売数

        # まだ購入していないエージェントだけを対象に行動
        non_holder_agents = self.participants.select(lambda agent: not agent.has_token)
        
        # 興味を持つかどうかを判定（確率的）
        non_holder_agents.shuffle_do("flag_if_interested", k=k)

        # フラグが立っているエージェントが購入を試みる
        non_holder_agents.shuffle_do("buy_if_flagged")
        
        # 結果をデータ収集
        self.datacollector.collect(self)
