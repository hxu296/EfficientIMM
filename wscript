#!/usr/bin/env python
# encoding: utf-8

# (the rest of the file remains unchanged)

VERSION = '0.0.1'
APPNAME = 'efficientimm'

def options(opt):
    opt.load('compiler_cxx')
    opt.load('waf_unit_test')
    opt.load('sphinx', tooldir='waftools')

    cfg_options = opt.get_option_group('Configuration options')

    opt.load('trng4', tooldir='waftools')
    opt.load('libjson', tooldir='waftools')
    opt.load('spdlog', tooldir='waftools')

    cfg_options.add_option(
        '--openmp-root', action='store', default='/usr',
        help='root directory of the installation of openmp')

    cfg_options.add_option(
        '--enable-mpi', action='store_true', default=False,
        help='enable openmpi implementation')

    cfg_options.add_option(
        '--enable-cuda', action='store_true', default=False,
        help='enable cuda implementation')
    
    cfg_options.add_option(
        '--build-ripples', action='store_true', default=False,
        help='build ripples version of the code')
    
    # Comment out papi
    # cfg_options.add_option(
    #     '--papi-root', action='store', default='/opt/cray/pe/papi/7.0.1.2/',
    #     help='root directory of the installation of PAPI')

    opt.load('mpi', tooldir='waftools')
    opt.load('cuda', tooldir='waftools')
    opt.load('memkind', tooldir='waftools')
    opt.load('metall', tooldir='waftools')
    

def configure(conf):
    try:
        build_dir = conf.options.out if conf.options.out != '' else 'build'
        conf.load('waf_conan_libs_info', tooldir=[build_dir, '.'])
    except:
        pass

    conf.load('compiler_cxx')
    conf.load('clang_compilation_database')
    conf.load('waf_unit_test')
    conf.load('sphinx', tooldir='waftools')

    if conf.options.enable_metall or conf.options.enable_cuda:
        conf.env.CXXFLAGS += ['-std=c++17', '-pipe']
    else:
        conf.env.CXXFLAGS += ['-std=c++17', '-pipe']

    conf.load('spdlog', tooldir='waftools')
    conf.load('libjson', tooldir='waftools')
    conf.load('trng4', tooldir='waftools')
    conf.load('catch2', tooldir='waftools')
    conf.load('cli', tooldir='waftools')

    conf.check_cxx(cxxflags=['-fopenmp'], ldflags=['-fopenmp'],
                   libpath=['{0}'.format(conf.options.openmp_root)],
                   uselib_store='OpenMP')

    # Hanjiang: Added the extra lib
    conf.check(lib='numa', uselib_store='NUMA')   

    # Comment out papi
    # # papi_root = conf.options.papi_root
    # conf.env.INCLUDES += ['{0}/include/'.format(conf.options.papi_root)]
    # conf.env.LIBPATH += ['{0}/lib/'.format(conf.options.papi_root)]
    # # conf.check_cxx(lib='trng4', uselib_store='libtrng',
    # #             includes=['{0}/include/'.format(conf.options.trng4_root)],
    # #             libpath=['{0}/lib/'.format(conf.options.trng4_root)])
    # conf.check_cxx(lib='papi', uselib_store='PAPI',
    #            includes=['{0}/include/'.format(conf.options.papi_root)],
    #            libpath=['{0}/lib/'.format(conf.options.papi_root)],
    #            header_name='papi.h')

    if conf.options.enable_mpi:
        conf.load('mpi', tooldir='waftools')

    conf.env.ENABLE_CUDA=False
    if conf.options.enable_cuda:
        conf.load('cuda', tooldir='waftools')
        conf.env.ENABLE_CUDA = True
        conf.env.CUDAFLAGS = ['--expt-relaxed-constexpr']

    if conf.options.enable_memkind and conf.options.enable_metall:
        conf.error('Metall and Memkind are mutually exclusive')

    conf.env.ENABLE_MEMKIND=False
    if conf.options.enable_memkind:
        conf.load('memkind', tooldir='waftools')
        conf.env.ENABLE_MEMKIND=True

    conf.env.ENABLE_METALL=False
    if conf.options.enable_metall:
        conf.load('metall', tooldir='waftools')
        conf.env.ENABLE_METALL=True


    env = conf.env
    conf.setenv('release', env)
    conf.env.append_value('CXXFLAGS', ['-O3', '-mtune=native'])
    # conf.env.append_value('LINKFLAGS', ['-lnuma', '-lpapi'])
    conf.env.append_value('LINKFLAGS', ['-lnuma'])
    
    conf.setenv('debug', env)
    conf.env.append_value('CXXFLAGS', ['-g', '-DDEBUG'])
    conf.env.append_value('LINKFLAGS', ['-lnuma', '-ltbb'])
    if conf.env.CXX == 'clang++':
        conf.env.append_value('CXXFLAGS', ['-O1', '-fsanitize=address', '-fno-omit-frame-pointer'])
    conf.env.append_value('CUDAFLAGS', ['-DDEBUG'])


def build(bld):
    if not bld.variant:
        bld.fatal('call "./waf build_release" or "./waf build_debug", and try "./waf --help"')

    directories = ['include', 'tools']

    if bld.options.build_ripples:
        directories = ['src_ripples/' + d for d in directories]
    else:
        directories = ['src_efficient_imm/' + d for d in directories]

    bld.recurse(directories)

    from waflib.Tools import waf_unit_test
    bld.add_post_fun(waf_unit_test.summary)


def build_docs(bld):
    if bld.env.ENABLE_DOCS:
        bld(features='sphinx', sources='docs')
    else:
        bld.fatal('Please configure with --enable-docs')


from waflib import Build
class docs(Build.BuildContext):
    fun = 'build_docs'
    cmd = 'docs'


from waflib.Build import BuildContext, CleanContext, InstallContext, UninstallContext
for x in 'debug release'.split():
    for y in (BuildContext, CleanContext, InstallContext, UninstallContext):
        name = y.__name__.replace('Context', '').lower()
        class tmp(y):
            cmd = name + '_' + x
            variant = x
