import os
import math
import random
import numpy as np
import pygame

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import scipy.sparse as sp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


LARGURA, ALTURA = 960, 600
FPS = 60

N_SENSORES = 14
FOV_GRAUS = 170
FOV_RENDER_GRAUS = 82
DIST_SENSOR = 280
DIST_MAX_RENDER = 900

NUM_RAIOS = LARGURA // 2
ALTURA_PAREDE_ESCALA = 280.0

PASSOS_SIMULACAO_POR_FRAME = 4
DECAY = 0.88
GANHO_ENTRADA = 5.0
RAIO_ESPECTRAL_ALVO = 0.72
GANHO_SAIDA_MOTORA = 160.0
SUAVIZACAO_VIRADA = 0.78

PESO_BUSCA_BASE = 1.35
PESO_DESVIO_BASE = 0.35


TAXA_APRENDIZADO = 0.08
DECAIMENTO_ELEGIBILIDADE = 0.96

PHOTORECEPTOR_PREFIXOS = ("R1-R6", "R7", "R8")
MOTOR_PREFIXO = "DNa"

CAMINHO_NEURONS = "neurons.csv"
CAMINHO_CONNECTIONS = "connections_princeton.csv"

COL_ID = "Root ID"
COL_TIPO = "Primary Cell Type"
COL_PRE = "pre_root_id"
COL_POS = "post_root_id"
COL_PESO = "syn_count"
COL_LADO = "Soma side"
COL_NT = "Predicted NT type"

SINAL_NT = {
    "ACH": 1.0, "GABA": -1.0, "GLUT": -1.0, "HIST": -1.0,
    "DA": 0.0, "SER": 0.0, "OCT": 0.0, "": 0.0,
}


class CerebroConectoma:
    def __init__(self, n_sensores):
        self.n_sensores = n_sensores
        self.usando_dados_reais = (
            os.path.exists(CAMINHO_NEURONS) and os.path.exists(CAMINHO_CONNECTIONS)
        )
        if self.usando_dados_reais:
            self._carregar_conectoma_real()
        else:
            self._gerar_conectoma_sintetico()

        self.v = np.zeros(self.n_neuronios, dtype=np.float32)
        self.spikes = np.zeros(self.n_neuronios, dtype=np.float32)
        self.virada_suave = 0.0

    def _recalibrar_raio_espectral(self):
        if not HAS_SCIPY or getattr(self.W, "nnz", 0) == 0:
            return
        try:
            raio = abs(
                sp.linalg.eigs(
                    self.W.astype(np.float64), k=1, which="LM", return_eigenvectors=False
                )[0]
            )
        except Exception:
            return
        if raio > 1e-6:
            self.W = self.W.multiply(RAIO_ESPECTRAL_ALVO / raio).tocsr()
            print(f"[cerebro] raio espectral bruto={raio:.2f} -> {RAIO_ESPECTRAL_ALVO}")

    def _carregar_conectoma_real(self):
        if not HAS_PANDAS or not HAS_SCIPY:
            raise RuntimeError("Precisa de pandas e scipy para o conectoma real.")

        print("[cerebro] lendo neurons.csv ...")
        df = pd.read_csv(
            CAMINHO_NEURONS,
            usecols=[COL_ID, COL_TIPO, COL_LADO, COL_NT],
            dtype={COL_ID: np.int64, COL_TIPO: "string", COL_LADO: "string", COL_NT: "string"},
        )
        ids = df[COL_ID].tolist()
        tipos = df[COL_TIPO].fillna("").tolist()
        lados = df[COL_LADO].fillna("").tolist()
        nts = df[COL_NT].fillna("").tolist()
        id_para_indice = {nid: i for i, nid in enumerate(ids)}
        self.n_neuronios = len(ids)
        sinal_por_id = {nid: SINAL_NT.get(nt, 0.0) for nid, nt in zip(ids, nts)}

        print(f"[cerebro] lendo {CAMINHO_CONNECTIONS} ...")
        pesos_por_par = {}
        for pedaco in pd.read_csv(
            CAMINHO_CONNECTIONS,
            usecols=[COL_PRE, COL_POS, COL_PESO],
            dtype={COL_PRE: np.int64, COL_POS: np.int64, COL_PESO: np.float32},
            chunksize=1_000_000,
        ):
            agrupado = pedaco.groupby([COL_PRE, COL_POS], as_index=False)[COL_PESO].sum()
            for pre, pos, peso in agrupado.itertuples(index=False):
                chave = (pre, pos)
                pesos_por_par[chave] = pesos_por_par.get(chave, 0.0) + peso

        linhas_pre, linhas_pos, pesos = [], [], []
        for (pre, pos), peso in pesos_por_par.items():
            ip = id_para_indice.get(pre)
            iq = id_para_indice.get(pos)
            if ip is not None and iq is not None:
                sinal = sinal_por_id.get(pre, 0.0)
                if sinal != 0.0:
                    linhas_pre.append(ip)
                    linhas_pos.append(iq)
                    pesos.append(peso * sinal)

        self.W = sp.csr_matrix(
            (pesos, (linhas_pre, linhas_pos)),
            shape=(self.n_neuronios, self.n_neuronios),
            dtype=np.float32,
        )
        if self.W.nnz > 0:
            escala = np.abs(self.W.data).mean()
            self.W = self.W.multiply(1.0 / max(1e-6, escala)).tocsr()
        self._recalibrar_raio_espectral()

        self._indices_fotorreceptores = [
            i for i, t in enumerate(tipos) if t.startswith(PHOTORECEPTOR_PREFIXOS)
        ]
        self._indices_foto_esq = [i for i in self._indices_fotorreceptores if lados[i] == "left"]
        self._indices_foto_dir = [i for i in self._indices_fotorreceptores if lados[i] == "right"]
        self._indices_motores = {
            "esquerda": [
                i for i, (t, lado) in enumerate(zip(tipos, lados))
                if t.startswith(MOTOR_PREFIXO) and lado == "left"
            ],
            "direita": [
                i for i, (t, lado) in enumerate(zip(tipos, lados))
                if t.startswith(MOTOR_PREFIXO) and lado == "right"
            ],
        }
        print(
            f"[cerebro] REAL: {self.n_neuronios} neuronios, {self.W.nnz} conexoes | "
            f"foto: {len(self._indices_fotorreceptores)} | "
            f"motores: { {k: len(v) for k, v in self._indices_motores.items()} }"
        )

    def _gerar_conectoma_sintetico(self):
        random.seed(42)
        np.random.seed(42)
        n_foto = self.n_sensores * 4
        n_meio = 300
        n_motor_esq, n_motor_dir = 16, 16
        self.n_neuronios = n_foto + n_meio + n_motor_esq + n_motor_dir

        self._indices_fotorreceptores = list(range(n_foto))
        self._indices_foto_esq = list(range(n_foto // 2))
        self._indices_foto_dir = list(range(n_foto // 2, n_foto))
        idx_meio = list(range(n_foto, n_foto + n_meio))
        idx_motor_esq = list(range(n_foto + n_meio, n_foto + n_meio + n_motor_esq))
        idx_motor_dir = list(range(n_foto + n_meio + n_motor_esq, self.n_neuronios))
        self._indices_motores = {"esquerda": idx_motor_esq, "direita": idx_motor_dir}

        linhas_pre, linhas_pos, pesos = [], [], []
        for f in self._indices_fotorreceptores:
            for a in random.sample(idx_meio, k=10):
                linhas_pre.append(f)
                linhas_pos.append(a)
                pesos.append(random.uniform(0.4, 1.0))
        for i, m in enumerate(idx_meio):
            alvos = random.sample(idx_motor_dir if i % 2 == 0 else idx_motor_esq, k=3)
            for a in alvos:
                linhas_pre.append(m)
                linhas_pos.append(a)
                pesos.append(random.uniform(0.5, 1.2))
        for m in idx_meio:
            for a in random.sample(idx_meio, k=2):
                if a != m:
                    linhas_pre.append(m)
                    linhas_pos.append(a)
                    pesos.append(random.uniform(-0.2, 0.2))

        if HAS_SCIPY:
            self.W = sp.csr_matrix(
                (pesos, (linhas_pre, linhas_pos)),
                shape=(self.n_neuronios, self.n_neuronios),
                dtype=np.float32,
            )
        else:
            self.W = np.zeros((self.n_neuronios, self.n_neuronios), dtype=np.float32)
            for pre, pos, w in zip(linhas_pre, linhas_pos, pesos):
                self.W[pre, pos] = w
        self._recalibrar_raio_espectral()
        print(f"[cerebro] sintetica ({self.n_neuronios} neuronios)")

    def passo(self, estimulo_esquerda, estimulo_direita):
        corrente = np.zeros(self.n_neuronios, dtype=np.float32)
        if self._indices_foto_esq:
            corrente[self._indices_foto_esq] = estimulo_esquerda * GANHO_ENTRADA
        if self._indices_foto_dir:
            corrente[self._indices_foto_dir] = estimulo_direita * GANHO_ENTRADA
        rede = self.spikes @ self.W
        if HAS_SCIPY and sp.issparse(self.W):
            rede = np.asarray(rede).flatten()
        self.v = self.v * DECAY + rede + corrente
        self.spikes = np.tanh(self.v)

    def leitura_motora(self):
        esq = (
            self.spikes[self._indices_motores["esquerda"]].mean()
            if self._indices_motores["esquerda"] else 0.0
        )
        dir_ = (
            self.spikes[self._indices_motores["direita"]].mean()
            if self._indices_motores["direita"] else 0.0
        )
        bruta = float(np.clip((dir_ - esq) * GANHO_SAIDA_MOTORA, -2.5, 2.5))
        self.virada_suave = SUAVIZACAO_VIRADA * self.virada_suave + (1 - SUAVIZACAO_VIRADA) * bruta
        return self.virada_suave

    def reset(self):
        self.v[:] = 0
        self.spikes[:] = 0
        self.virada_suave = 0.0


class Aprendiz:
    def __init__(self):
        self.w_desvio = PESO_DESVIO_BASE
        self.w_busca = PESO_BUSCA_BASE
        self.eleg_desvio = 0.0
        self.eleg_busca = 0.0
        self.historico_batidas = []

    def passo_acao(self, desvio, busca):
        self.eleg_desvio = DECAIMENTO_ELEGIBILIDADE * self.eleg_desvio + abs(desvio)
        self.eleg_busca = DECAIMENTO_ELEGIBILIDADE * self.eleg_busca + abs(busca)

    def reforco(self, recompensa):
        self.w_desvio += TAXA_APRENDIZADO * recompensa * self.eleg_desvio * (-1.0 if recompensa < 0 else 0.15)
        if recompensa < 0:
            self.w_desvio += TAXA_APRENDIZADO * abs(recompensa) * (0.5 + self.eleg_desvio)
            self.w_busca -= TAXA_APRENDIZADO * 0.05 * abs(recompensa)
        else:
            self.w_busca += TAXA_APRENDIZADO * recompensa * (0.4 + self.eleg_busca)
            self.w_desvio += TAXA_APRENDIZADO * 0.05 * recompensa

        self.w_desvio = float(np.clip(self.w_desvio, 0.15, 1.4))
        self.w_busca = float(np.clip(self.w_busca, 0.4, 2.6))

    def reset(self):
        self.w_desvio = PESO_DESVIO_BASE
        self.w_busca = PESO_BUSCA_BASE
        self.eleg_desvio = 0.0
        self.eleg_busca = 0.0
        self.historico_batidas.clear()


def sensores_paredes(pos, angulo, paredes, n_sensores):
    estimulo = np.zeros(n_sensores, dtype=np.float32)
    meio = math.radians(FOV_GRAUS) / 2
    for i in range(n_sensores):
        frac = i / max(1, n_sensores - 1)
        ang = angulo - meio + frac * 2 * meio
        dx, dy = math.cos(ang), math.sin(ang)
        menor = DIST_SENSOR
        for px, py, pw, ph in paredes:
            for t in np.linspace(0, 1, 7):
                for bx, by in (
                    (px, py + t * ph),
                    (px + pw, py + t * ph),
                    (px + t * pw, py),
                    (px + t * pw, py + ph),
                ):
                    vx, vy = bx - pos[0], by - pos[1]
                    proj = vx * dx + vy * dy
                    if 0 < proj < DIST_SENSOR:
                        perp = abs(vx * dy - vy * dx)
                        if perp < 16:
                            menor = min(menor, proj)
        estimulo[i] = max(0.0, 1.0 - menor / DIST_SENSOR)
    metade = n_sensores // 2
    return float(estimulo[:metade].max()), float(estimulo[metade:].max())


def direcao_para_alvo(pos, angulo, alvo):
    dx = alvo[0] - pos[0]
    dy = alvo[1] - pos[1]
    ang_alvo = math.atan2(dy, dx)
    diff = (ang_alvo - angulo + math.pi) % (2 * math.pi) - math.pi
    return float(np.clip(diff / (math.pi * 0.6), -1.0, 1.0))


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def ray_vs_rect(px, py, dx, dy, rx, ry, rw, rh):
    inv_dx = 1e9 if dx == 0 else 1.0 / dx
    inv_dy = 1e9 if dy == 0 else 1.0 / dy

    tx1 = (rx - px) * inv_dx
    tx2 = (rx + rw - px) * inv_dx
    tmin, tmax = min(tx1, tx2), max(tx1, tx2)

    ty1 = (ry - py) * inv_dy
    ty2 = (ry + rh - py) * inv_dy
    tmin = max(tmin, min(ty1, ty2))
    tmax = min(tmax, max(ty1, ty2))

    if tmax >= max(tmin, 0):
        return tmin if tmin > 0 else tmax
    return None


def lancar_raios(pos, angulo, paredes, num_raios, fov_graus):
    meio = math.radians(fov_graus) / 2
    distancias = []
    for i in range(num_raios):
        frac = i / max(1, num_raios - 1)
        ang = angulo - meio + frac * 2 * meio
        dx, dy = math.cos(ang), math.sin(ang)
        menor = DIST_MAX_RENDER
        for (px, py, pw, ph) in paredes:
            t = ray_vs_rect(pos[0], pos[1], dx, dy, px, py, pw, ph)
            if t is not None and 0 < t < menor:
                menor = t

        perp = menor * math.cos(ang - angulo)
        distancias.append(max(0.0001, perp))
    return distancias


def desenhar_primeira_pessoa(tela, pos, angulo, paredes, distancias, alvo_comida, alvo_ninho, carregando):
    largura_col = LARGURA / len(distancias)

    tela.fill((30, 30, 45), (0, 0, LARGURA, ALTURA // 2))
    tela.fill((22, 20, 18), (0, ALTURA // 2, LARGURA, ALTURA // 2))
    for k in range(8):
        fase = (pos[0] * math.cos(angulo) + pos[1] * math.sin(angulo)) * 0.04
        y = int(ALTURA * 0.52 + k * 18 + (fase % 18))
        if ALTURA // 2 < y < ALTURA:
            tom = 22 + k * 2
            pygame.draw.line(tela, (tom, max(0, tom - 2), max(0, tom - 4)), (0, y), (LARGURA, y), 1)


    for i, d in enumerate(distancias):
        altura_col = min(ALTURA, (ALTURA_PAREDE_ESCALA * ALTURA) / (d + 1e-6) / 1.6)
        sombra = max(0.15, 1.0 - d / DIST_MAX_RENDER)
        cor = (int(150 * sombra) + 40, int(60 * sombra) + 15, int(60 * sombra) + 15)
        x = int(i * largura_col)
        y = int((ALTURA - altura_col) / 2)
        pygame.draw.rect(tela, cor, (x, y, math.ceil(largura_col) + 1, int(altura_col)))

    for nome, alvo, cor_base, ativo in (
        ("COMIDA", alvo_comida, (255, 170, 50), not carregando),
        ("NINHO", alvo_ninho, (80, 220, 255), True),
    ):
        if not ativo:
            continue
        dx = alvo[0] - pos[0]
        dy = alvo[1] - pos[1]
        d_sprite = math.hypot(dx, dy)
        ang_alvo = math.atan2(dy, dx)
        diff = (ang_alvo - angulo + math.pi) % (2 * math.pi) - math.pi
        meio_fov = math.radians(FOV_RENDER_GRAUS) / 2
        if abs(diff) > meio_fov * 1.15 or d_sprite < 1:
            continue
        col_centro = (0.5 + diff / (2 * meio_fov)) * LARGURA
        idx_col = int(np.clip(col_centro / largura_col, 0, len(distancias) - 1))
        if d_sprite > distancias[idx_col] + 25:
            continue
        tam = float(np.clip((ALTURA_PAREDE_ESCALA * ALTURA) / (d_sprite + 8.0) / 2.2, 10, 280))
        cy = ALTURA * 0.52 + min(90, d_sprite * 0.08)
        if d_sprite < 80:
            cy = ALTURA * 0.58
        if d_sprite < 40:
            cy = ALTURA * 0.66
        r = int(tam / 2)
        pygame.draw.circle(tela, cor_base, (int(col_centro), int(cy)), r)
        pygame.draw.circle(tela, (255, 255, 255), (int(col_centro), int(cy)), r, 2)
        if d_sprite < 55:
            pygame.draw.circle(tela, (255, 255, 180), (int(col_centro), int(cy)), r + 6, 2)
        if d_sprite < 130:
            fs = pygame.font.SysFont("consolas", max(12, int(13 + (90 - min(90, d_sprite)) * 0.12)))
            lab = fs.render(f"{nome} {d_sprite:.0f}", True, cor_base)
            tela.blit(lab, (int(col_centro) - lab.get_width() // 2, int(cy) - r - 18))


    pygame.draw.line(tela, (200, 200, 200), (LARGURA // 2 - 6, ALTURA // 2), (LARGURA // 2 + 6, ALTURA // 2), 1)
    pygame.draw.line(tela, (200, 200, 200), (LARGURA // 2, ALTURA // 2 - 6), (LARGURA // 2, ALTURA // 2 + 6), 1)


def desenhar_minimapa(tela, pos, angulo, paredes, comida, ninho, carregando, trilha):
    escala = 0.16
    ox, oy = LARGURA - int(LARGURA * escala) - 12, 12
    largura_mm, altura_mm = int(LARGURA * escala), int(ALTURA * escala)

    fundo = pygame.Surface((largura_mm, altura_mm))
    fundo.fill((10, 10, 16))
    for px, py, pw, ph in paredes:
        pygame.draw.rect(fundo, (150, 60, 60), (px * escala, py * escala, pw * escala, ph * escala))
    if len(trilha) > 1:
        pontos = [(int(x * escala), int(y * escala)) for x, y in trilha]
        pygame.draw.lines(fundo, (35, 90, 110), False, pontos, 1)
    if not carregando:
        pygame.draw.circle(fundo, (255, 170, 50), (int(comida[0] * escala), int(comida[1] * escala)), 3)
    pygame.draw.circle(fundo, (80, 220, 255), (int(ninho[0] * escala), int(ninho[1] * escala)), 3)

    px_, py_ = int(pos[0] * escala), int(pos[1] * escala)
    ponta = (px_ + math.cos(angulo) * 6, py_ + math.sin(angulo) * 6)
    e = (px_ + math.cos(angulo + 2.4) * 4, py_ + math.sin(angulo + 2.4) * 4)
    d = (px_ + math.cos(angulo - 2.4) * 4, py_ + math.sin(angulo - 2.4) * 4)
    pygame.draw.polygon(fundo, (255, 255, 0), [ponta, e, d])

    tela.blit(fundo, (ox, oy))
    pygame.draw.rect(tela, (100, 100, 120), (ox, oy, largura_mm, altura_mm), 1)


def main():
    pygame.init()
    tela = pygame.display.set_mode((LARGURA, ALTURA))
    pygame.display.set_caption("Moscardo DOOM - bordas + sprites + giro melhor")
    relogio = pygame.time.Clock()
    fonte = pygame.font.SysFont("consolas", 17)

    cerebro = CerebroConectoma(N_SENSORES)
    aprendiz = Aprendiz()

    pos = [120.0, ALTURA / 2]
    angulo = 0.0
    velocidade = 145.0

    ESP = 30
    paredes = [
        [0, 0, LARGURA, ESP],
        [0, ALTURA - ESP, LARGURA, ESP],
        [0, 0, ESP, ALTURA],
        [LARGURA - ESP, 0, ESP, ALTURA],
        [280, 80, 40, 200],
        [280, 360, 40, 180],
        [480, ESP, 40, 200],
        [480, 340, 40, 260],
        [680, 100, 40, 160],
        [680, 380, 40, 180],
    ]

    comida = [720.0, 160.0]
    ninho = [100.0, 480.0]
    carregando = False
    entregas = 0
    batidas = 0
    tempo = 0.0
    trilha = []
    rodando = True
    mostrar_minimapa = True

    boost_esq = 0.0
    boost_dir = 0.0
    dor = 0.0
    dopamina = 0.0
    choque = 0.0
    parado_mult = 1.0
    livre_arb = 0.0
    tempo_parado = 0.0
    quer_parar = False
    cooldown_colisao = 0.0
    flash_pickup = 0.0

    while rodando:
        dt = relogio.tick(FPS) / 1000.0
        tempo += dt
        boost_esq *= 0.92
        boost_dir *= 0.92
        dor = max(0.0, dor - 1.2 * dt)
        dopamina = max(0.0, dopamina - 0.7 * dt)
        choque = max(0.0, choque - 2.5 * dt)
        livre_arb *= 0.90
        flash_pickup = max(0.0, flash_pickup - 1.8 * dt)

        if not quer_parar and random.random() < 0.004:
            quer_parar = True
            tempo_parado = random.uniform(0.4, 2.2)
        if quer_parar:
            tempo_parado -= dt
            parado_mult = 0.05
            if tempo_parado <= 0 or choque > 0.1:
                quer_parar = False
                parado_mult = 1.0
        else:
            parado_mult = 1.0
            if random.random() < 0.02:
                livre_arb += random.uniform(-0.9, 0.9)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                rodando = False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    rodando = False
                if ev.key == pygame.K_LEFT:
                    boost_esq = 1.0
                if ev.key == pygame.K_RIGHT:
                    boost_dir = 1.0
                if ev.key == pygame.K_TAB:
                    mostrar_minimapa = not mostrar_minimapa
                if ev.key == pygame.K_r:
                    aprendiz.reset()
                    print("[aprendizado] pesos resetados")
                if ev.key == pygame.K_c:
                    choque = 1.0
                    quer_parar = False
                    parado_mult = 1.0
                    livre_arb += random.uniform(-1.5, 1.5)
                    cerebro.v += np.random.uniform(-0.5, 0.5, size=cerebro.n_neuronios).astype(np.float32)
                    print("[choque]")
                if ev.key == pygame.K_SPACE:
                    cerebro.reset()
                    pos = [120.0, ALTURA / 2]
                    angulo = 0.0
                    carregando = False
                    dor = 0.0
                    dopamina = 0.0
                    trilha.clear()
                    comida = [random.uniform(500, 820), random.uniform(80, 520)]

        est_esq, est_dir = sensores_paredes(pos, angulo, paredes, N_SENSORES)
        est_esq = min(1.0, est_esq + boost_esq)
        est_dir = min(1.0, est_dir + boost_dir)

        for _ in range(PASSOS_SIMULACAO_POR_FRAME):
            cerebro.passo(est_esq, est_dir)
        virada_cerebro = cerebro.leitura_motora()

        alvo = ninho if carregando else comida
        busca = direcao_para_alvo(pos, angulo, alvo)
        desvio = (est_esq - est_dir) * 2.0

        aprendiz.passo_acao(desvio, busca)

        virada = (
            aprendiz.w_desvio * desvio * (1.0 + 0.45 * dor)
            + aprendiz.w_busca * busca * (1.25 + 0.7 * dopamina)
            + 0.25 * virada_cerebro
            + livre_arb
        )
        virada = float(np.clip(virada, -2.8, 2.8))

        vel_atual = velocidade * (1.0 - 0.4 * dor) * (1.0 + 0.35 * dopamina) * parado_mult
        if choque > 0.05:
            vel_atual = max(vel_atual, velocidade * (0.9 + 1.2 * choque))
            virada += random.uniform(-1.2, 1.2) * choque
        vel_atual = max(0.0, vel_atual)

        angulo += virada * 2.8 * dt
        pos[0] += math.cos(angulo) * vel_atual * dt
        pos[1] += math.sin(angulo) * vel_atual * dt

        raio_mosca = 9
        bateu = False
        for px, py, pw, ph in paredes:
            left, right = px - raio_mosca, px + pw + raio_mosca
            top, bottom = py - raio_mosca, py + ph + raio_mosca
            if not (left < pos[0] < right and top < pos[1] < bottom):
                continue
            bateu = True
            dl, dr = pos[0] - left, right - pos[0]
            dtp, db = pos[1] - top, bottom - pos[1]
            menor = min(dl, dr, dtp, db)
            extra = 4.0
            if menor == dl:
                pos[0] = left - extra
                angulo = math.atan2(math.sin(angulo), -abs(math.cos(angulo)))
            elif menor == dr:
                pos[0] = right + extra
                angulo = math.atan2(math.sin(angulo), abs(math.cos(angulo)))
            elif menor == dtp:
                pos[1] = top - extra
                angulo = math.atan2(-abs(math.sin(angulo)), math.cos(angulo))
            else:
                pos[1] = bottom + extra
                angulo = math.atan2(abs(math.sin(angulo)), math.cos(angulo))
            angulo += random.uniform(-0.12, 0.12)

        if bateu and cooldown_colisao <= 0:
            dor = min(1.0, dor + 0.55)
            batidas += 1
            aprendiz.reforco(-1.0)
            cerebro.v += np.random.uniform(-0.2, 0.2, size=cerebro.n_neuronios).astype(np.float32)
            cooldown_colisao = 0.35
        if cooldown_colisao > 0:
            cooldown_colisao -= dt

        angulo = (angulo + math.pi) % (2 * math.pi) - math.pi

        if not carregando and dist(pos, comida) < 22:
            carregando = True
            dopamina = min(1.0, dopamina + 0.85)
            dor *= 0.3
            aprendiz.reforco(+0.8)
            flash_pickup = 0.65
            print("[jogo] Comida!")

        if carregando and dist(pos, ninho) < 28:
            carregando = False
            entregas += 1
            dopamina = 1.0
            dor = 0.0
            aprendiz.reforco(+1.2)
            comida = [random.uniform(500, 820), random.uniform(80, 520)]
            flash_pickup = 0.85
            print(f"[jogo] Entrega #{entregas}")

        trilha.append((int(pos[0]), int(pos[1])))
        if len(trilha) > 300:
            trilha.pop(0)


        distancias = lancar_raios(pos, angulo, paredes, NUM_RAIOS, FOV_RENDER_GRAUS)
        desenhar_primeira_pessoa(tela, pos, angulo, paredes, distancias, comida, ninho, carregando)

        if flash_pickup > 0:
            s = pygame.Surface((LARGURA, ALTURA), pygame.SRCALPHA)
            s.fill((255, 220, 80, int(130 * flash_pickup)))
            tela.blit(s, (0, 0))

        if mostrar_minimapa:
            desenhar_minimapa(tela, pos, angulo, paredes, comida, ninho, carregando, trilha)


        def barra(x, y, valor, cor_cheia, label):
            pygame.draw.rect(tela, (40, 40, 50), (x, y, 120, 12))
            pygame.draw.rect(tela, cor_cheia, (x, y, int(120 * min(1.0, valor)), 12))
            pygame.draw.rect(tela, (100, 100, 120), (x, y, 120, 12), 1)
            tela.blit(fonte.render(label, True, (220, 220, 220)), (x + 125, y - 2))

        barra(10, 12, dor, (220, 60, 60), f"DOR {dor:.2f}")
        barra(10, 30, dopamina, (240, 200, 50), f"DA  {dopamina:.2f}")
        barra(10, 48, choque, (255, 100, 255), f"CHOQUE {choque:.2f}")
        barra(10, 66, (aprendiz.w_desvio - 0.3) / 2.2, (100, 180, 255), f"w_desvio {aprendiz.w_desvio:.2f}")
        barra(10, 84, (aprendiz.w_busca - 0.15) / 1.85, (100, 255, 160), f"w_busca  {aprendiz.w_busca:.2f}")

        modo = "CONECTOMA REAL" if cerebro.usando_dados_reais else "sintetico"
        estado = "carregando -> ninho" if carregando else "buscando comida"
        txt1 = fonte.render(f"cerebro: {modo}  |  neuronios: {cerebro.n_neuronios}", True, (230, 230, 230))
        txt2 = fonte.render(
            f"{estado}  |  entregas: {entregas}  batidas: {batidas}  t: {tempo:.0f}s",
            True, (230, 230, 230),
        )
        txt3 = fonte.render(
            "SETAS=estimulo  C=choque  ESPACO=reset pos  R=reset aprend  TAB=mapa  ESC=sair",
            True, (200, 200, 200),
        )
        tela.blit(txt1, (10, ALTURA - 46))
        tela.blit(txt2, (10, ALTURA - 28))
        tela.blit(txt3, (10, ALTURA - 12))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
