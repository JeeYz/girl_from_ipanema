print('this is test...')




import pygame

# 초기화
pygame.init()

# 창 크기 설정
size = (700, 500)
screen = pygame.display.set_mode(size)

# 이미지 로드
dot_image = pygame.image.load("dot.png").convert_alpha()

# 도트 객체 생성
class Dot(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = dot_image
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.direction = "right"

    def update(self):
        if self.direction == "right":
            self.rect.x += 5
            if self.rect.x >= 650:
                self.direction = "down"
        elif self.direction == "down":
            self.rect.y += 5
            if self.rect.y >= 450:
                self.direction = "left"
        elif self.direction == "left":
            self.rect.x -= 5
            if self.rect.x <= 0:
                self.direction = "up"
        elif self.direction == "up":
            self.rect.y -= 5
            if self.rect.y <= 0:
                self.direction = "right"

# 도트 그룹 생성
all_sprites_list = pygame.sprite.Group()

# 도트 객체 생성 및 그룹에 추가
dot = Dot(0, 0)
all_sprites_list.add(dot)

# 게임 루프
done = False
clock = pygame.time.Clock()

while not done:
    # 이벤트 처리
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    # 화면 지우기
    screen.fill((255, 255, 255))

    # 도트 그리기
    all_sprites_list.update()
    all_sprites_list.draw(screen)

    # 화면 업데이트
    pygame.display.flip()

    # 게임 속도 조절
    clock.tick(60)

# 게임 종료
pygame.quit()


