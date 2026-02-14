"""
API认证测试

测试JWT认证、用户登录注册等功能
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient
from src.api.main import app


client = TestClient(app)


def test_health_check():
    """测试健康检查接口（无需认证）"""
    print("=" * 60)
    print("测试健康检查接口")
    print("=" * 60)

    response = client.get("/health")
    print(f"  状态码: {response.status_code}")
    print(f"  响应: {response.json()}")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("  ✅ 健康检查测试通过")


def test_login():
    """测试用户登录"""
    print("\n" + "=" * 60)
    print("测试用户登录")
    print("=" * 60)

    # 使用预设的admin账户
    response = client.post(
        "/auth/login",
        data={
            "username": "admin",
            "password": "admin123",
        }
    )

    print(f"  状态码: {response.status_code}")
    data = response.json()
    print(f"  token_type: {data.get('token_type')}")
    print(f"  expires_in: {data.get('expires_in')}")

    assert response.status_code == 200
    assert "access_token" in data
    assert data["token_type"] == "bearer"

    print("  ✅ 登录测试通过")
    return data["access_token"]


def test_login_invalid():
    """测试登录失败"""
    print("\n" + "=" * 60)
    print("测试登录失败")
    print("=" * 60)

    response = client.post(
        "/auth/login",
        data={
            "username": "admin",
            "password": "wrong_password",
        }
    )

    print(f"  状态码: {response.status_code}")

    assert response.status_code == 401
    print("  ✅ 错误密码测试通过")


def test_get_current_user():
    """测试获取当前用户"""
    print("\n" + "=" * 60)
    print("测试获取当前用户")
    print("=" * 60)

    # 先登录获取token
    token = test_login()

    # 获取用户信息
    response = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {token}"}
    )

    print(f"  状态码: {response.status_code}")
    data = response.json()
    print(f"  username: {data.get('username')}")
    print(f"  scopes: {data.get('scopes')}")

    assert response.status_code == 200
    assert data["username"] == "admin"

    print("  ✅ 获取用户信息测试通过")


def test_get_current_user_no_token():
    """测试无token访问"""
    print("\n" + "=" * 60)
    print("测试无token访问")
    print("=" * 60)

    response = client.get("/auth/me")

    print(f"  状态码: {response.status_code}")

    assert response.status_code == 401
    print("  ✅ 无token访问测试通过")


def test_verify_token():
    """测试token验证"""
    print("\n" + "=" * 60)
    print("测试token验证")
    print("=" * 60)

    # 先登录获取token
    token = test_login()

    # 验证token
    response = client.get(
        "/auth/verify",
        headers={"Authorization": f"Bearer {token}"}
    )

    print(f"  状态码: {response.status_code}")
    data = response.json()
    print(f"  valid: {data.get('valid')}")
    print(f"  username: {data.get('username')}")

    assert response.status_code == 200
    assert data["valid"] is True

    print("  ✅ Token验证测试通过")


def test_register():
    """测试用户注册"""
    print("\n" + "=" * 60)
    print("测试用户注册")
    print("=" * 60)

    # 尝试注册新用户
    import uuid
    unique_username = f"test_user_{uuid.uuid4().hex[:8]}"

    response = client.post(
        "/auth/register",
        json={
            "username": unique_username,
            "password": "test123456",
            "email": f"{unique_username}@test.com",
        }
    )

    print(f"  状态码: {response.status_code}")
    data = response.json()
    print(f"  success: {data.get('success')}")
    print(f"  message: {data.get('message')}")

    assert response.status_code == 200
    assert data["success"] is True

    # 尝试用新用户登录
    login_response = client.post(
        "/auth/login",
        data={
            "username": unique_username,
            "password": "test123456",
        }
    )

    print(f"  新用户登录状态码: {login_response.status_code}")
    assert login_response.status_code == 200

    print("  ✅ 用户注册测试通过")


def test_refresh_token():
    """测试刷新token"""
    print("\n" + "=" * 60)
    print("测试刷新token")
    print("=" * 60)

    # 先登录获取token
    token = test_login()

    # 刷新token
    response = client.post(
        "/auth/token/refresh",
        headers={"Authorization": f"Bearer {token}"}
    )

    print(f"  状态码: {response.status_code}")
    data = response.json()
    print(f"  token_type: {data.get('token_type')}")

    assert response.status_code == 200
    assert "access_token" in data
    assert data["token_type"] == "bearer"

    print("  ✅ 刷新token测试通过")


if __name__ == "__main__":
    test_health_check()
    test_login()
    test_login_invalid()
    test_get_current_user()
    test_get_current_user_no_token()
    test_verify_token()
    test_register()
    test_refresh_token()

    print("\n" + "=" * 60)
    print("所有认证测试通过 ✅")
    print("=" * 60)
