import axios from "axios";
import keycloak, { getTokenMinValidity } from "./keycloak";


const axiosInstance = axios.create(
{
    baseURL: "http://localhost:8090/api",
    withCredentials: true
})


axiosInstance.interceptors.request.use(async (config) =>
{
    await keycloak.updateToken(getTokenMinValidity());

    if (keycloak.token)
    {
        config.headers.Authorization = `Bearer ${keycloak.token}`;
    }

    return config;
})

export default axiosInstance;