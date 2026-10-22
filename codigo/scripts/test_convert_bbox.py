"""
Teste de validação da fórmula de conversão BDD100K → YOLO.
Verifica que as coordenadas normalizadas estão corretas.
"""

# Simular a função de conversão
def convert_bbox_to_yolo(x1, y1, x2, y2, img_w=1280, img_h=720):
    if x2 <= x1 or y2 <= y1:
        return None
    x_center = (x1 + x2) / (2.0 * img_w)
    y_center = (y1 + y2) / (2.0 * img_h)
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    return (x_center, y_center, w, h)

# ── Testes ────────────────────────────────────────────────────────
print("=" * 60)
print("  Validação da Fórmula de Conversão BDD100K → YOLO")
print("=" * 60)

# Teste 1: Carro no centro da imagem (640,360 é o centro de 1280x720)
result = convert_bbox_to_yolo(540, 260, 740, 460)
expected = (0.5, 0.5, 200/1280, 200/720)
assert result is not None
print(f"\n  Teste 1 — Bbox centralizada (540,260,740,460)")
print(f"    Resultado:  x={result[0]:.4f} y={result[1]:.4f} w={result[2]:.4f} h={result[3]:.4f}")
print(f"    Esperado:   x={expected[0]:.4f} y={expected[1]:.4f} w={expected[2]:.4f} h={expected[3]:.4f}")
assert abs(result[0] - expected[0]) < 1e-6, f"x_center errado: {result[0]} != {expected[0]}"
assert abs(result[1] - expected[1]) < 1e-6, f"y_center errado"
print(f"    ✅ PASSOU")

# Teste 2: Canto superior esquerdo
result = convert_bbox_to_yolo(0, 0, 128, 72)
expected = (0.05, 0.05, 0.1, 0.1)
print(f"\n  Teste 2 — Canto superior esquerdo (0,0,128,72)")
print(f"    Resultado:  x={result[0]:.4f} y={result[1]:.4f} w={result[2]:.4f} h={result[3]:.4f}")
print(f"    Esperado:   x={expected[0]:.4f} y={expected[1]:.4f} w={expected[2]:.4f} h={expected[3]:.4f}")
assert abs(result[0] - expected[0]) < 1e-6
assert abs(result[2] - expected[2]) < 1e-6
print(f"    ✅ PASSOU")

# Teste 3: Imagem inteira
result = convert_bbox_to_yolo(0, 0, 1280, 720)
expected = (0.5, 0.5, 1.0, 1.0)
print(f"\n  Teste 3 — Imagem inteira (0,0,1280,720)")
print(f"    Resultado:  x={result[0]:.4f} y={result[1]:.4f} w={result[2]:.4f} h={result[3]:.4f}")
assert abs(result[0] - 0.5) < 1e-6
assert abs(result[1] - 0.5) < 1e-6
assert abs(result[2] - 1.0) < 1e-6
assert abs(result[3] - 1.0) < 1e-6
print(f"    ✅ PASSOU")

# Teste 4: Bbox inválida (x2 < x1)
result = convert_bbox_to_yolo(500, 300, 400, 200)
assert result is None
print(f"\n  Teste 4 — Bbox inválida (x2 < x1)")
print(f"    Resultado:  None")
print(f"    ✅ PASSOU (corretamente rejeitada)")

# Teste 5: Bbox no canto inferior direito
result = convert_bbox_to_yolo(1180, 620, 1280, 720)
expected = ((1180+1280)/(2*1280), (620+720)/(2*720), 100/1280, 100/720)
print(f"\n  Teste 5 — Canto inferior direito (1180,620,1280,720)")
print(f"    Resultado:  x={result[0]:.6f} y={result[1]:.6f} w={result[2]:.6f} h={result[3]:.6f}")
print(f"    Esperado:   x={expected[0]:.6f} y={expected[1]:.6f} w={expected[2]:.6f} h={expected[3]:.6f}")
assert abs(result[0] - expected[0]) < 1e-6
assert abs(result[1] - expected[1]) < 1e-6
print(f"    ✅ PASSOU")

# Teste 6: Valores reais do BDD100K (carro típico)
result = convert_bbox_to_yolo(594.61, 349.05, 780.96, 445.58)
print(f"\n  Teste 6 — Valor real BDD100K (carro)")
print(f"    Input:      x1=594.61 y1=349.05 x2=780.96 y2=445.58")
print(f"    Resultado:  x={result[0]:.6f} y={result[1]:.6f} w={result[2]:.6f} h={result[3]:.6f}")
assert 0 <= result[0] <= 1 and 0 <= result[1] <= 1
assert 0 <= result[2] <= 1 and 0 <= result[3] <= 1
print(f"    ✅ PASSOU (todos os valores em [0,1])")

print(f"\n{'='*60}")
print(f"  ✅ TODOS OS 6 TESTES PASSARAM")
print(f"{'='*60}")
