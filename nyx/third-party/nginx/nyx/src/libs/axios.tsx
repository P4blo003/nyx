// ==========================================================================================
// Author: Pablo González García.
// Created: 19/02/2026
// Last edited: 20/02/2026
// ==========================================================================================


// ==============================
// IMPORTS
// ==============================

// External:
import axios from "axios";

// Internal:
import keycloak, { getTokenMinValidity } from "./keycloak";


// ==============================
// INSTANCES
// ==============================

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


// ==============================
// EXPORTS
// ==============================

export default axiosInstance;