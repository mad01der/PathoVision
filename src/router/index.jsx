import React, { Component } from 'react'
// 导入所需组件
import Login from '../components/login'
import Home from '../components/home'
import Load from '../components/load'
import Data from '../components/data_in'
import Home_load from '../components/home_load'
// 导入路由依赖
import { Route,BrowserRouter } from 'react-router-dom'
 
export default class index extends Component {
  render() {
    return (
        // 使用BrowserRouter包裹，配置路由
      <BrowserRouter>
         {/* 使用/配置路由默认页；exact严格匹配 */}
        <Route component={Login} path='/' exact></Route>
        <Route component={Login} path='/Login'></Route>
        <Route component={Load} path='/Load'></Route>
        <Route component={Home} path='/Home'></Route>
        <Route component={Data} path='/Data'></Route>
        <Route component={Home_load} path='/Home_load'></Route>
      </BrowserRouter>
    )
  }
}
